# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import os.path as osp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from tensorboardX import SummaryWriter

from utils.distribution import DistributionMetric, Sample
from utils.reid_metric import R1_mAP


def create_online_trainer(model, optimizer, loss_fn, train_loader, device=None):
    if device:
        model.to(device)

    def _update(engine, batch):
        t1, t2 = engine.threshold['t1'], engine.threshold['t2']  # get threshold
        # prepare triplet or pair samples
        imgs, labels = batch
        imgs = imgs.cuda()
        model.eval()
        with torch.no_grad():
            batch_feat = F.normalize(model(imgs))
        cosine_dist = torch.mm(batch_feat, engine.train_feats.t())  # (batch, 500)
        sorted_dist, sorted_idx = torch.sort(cosine_dist, dim=1)

        picked_index = []
        positive_pick = 0
        positive_total = (labels.unsqueeze(1) == engine.train_labels.unsqueeze(0)).sum().item()
        negative_pick = 0
        negative_total = (labels.unsqueeze(1) != engine.train_labels.unsqueeze(0)).sum().item()
        for i in range(sorted_dist.shape[0]):
            label = labels[i].item()
            p_idx = None
            n_idx = None
            d = sorted_dist[i]
            index = sorted_idx[i]
            # positive sample
            positive_sets = torch.masked_select(d, d > t2)
            positive_index = torch.masked_select(index, d > t2)
            if len(positive_sets) != 0:
                p_d = positive_sets[0]  # positive hard
                p_idx = torch.masked_select(index, d == p_d)[0].item()
                positive_pick += (label == engine.train_labels[positive_index]).float().sum().item()

            # negative sample
            negative_sets = torch.masked_select(d, d < t1)
            negative_index = torch.masked_select(index, d < t1)
            if len(negative_sets) != 0:
                n_d = negative_sets[-1]  # negative hard
                n_idx = torch.masked_select(index, d == n_d)[0].item()
                negative_pick += (label != engine.train_labels[negative_index]).float().sum().item()

            picked_index.append((i, p_idx, n_idx))

        # Get online batch positive/negative pair info
        pos_count = 0
        pos_correct = 0
        neg_count = 0
        neg_correct = 0
        for l_i, label in enumerate(labels):
            label = label.item()
            _, p_idx, n_idx = picked_index[l_i]
            if p_idx is not None:
                pos_count += 1
                if label == p_idx:
                    pos_correct += 1
            if n_idx is not None:
                neg_count += 1
                if label != n_idx:
                    neg_correct += 1

        model.train()
        optimizer.zero_grad()
        feats = model(imgs)
        train_feats = {}
        for p_i in picked_index:
            s_i, s_p, s_n = p_i
            if s_p is not None and s_p not in train_feats:
                img_tensor, _, _, _ = train_loader.dataset[s_p]
                img_tensor = img_tensor.unsqueeze(0).cuda()
                with torch.no_grad():
                    train_feats[s_p] = model(img_tensor)
            if s_n is not None and s_n not in train_feats:
                img_tensor, _, _, _ = train_loader.dataset[s_n]
                img_tensor = img_tensor.unsqueeze(0).cuda()
                with torch.no_grad():
                    train_feats[s_n] = model(img_tensor)

        loss = loss_fn(feats, train_feats, picked_index)
        if loss == 0:
            return engine.state.output
        else:
            loss.backward()
            optimizer.step()
            return loss.item(), (pos_count, pos_correct, neg_count, neg_correct), \
                   (positive_pick, positive_total, negative_pick, negative_total)

    return Engine(_update)


def create_supervised_evaluator(model, metrics, device=None):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        online_loader,
        val_loader,
        test_loader,
        optimizer,
        loss_fn,
        num_query
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    dist_period = cfg.SOLVER.DIST_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_online.train")
    logger.info("Start training")
    writer = SummaryWriter(osp.join(output_dir, 'board'))

    trainer = create_online_trainer(model, optimizer, loss_fn, train_loader, device=device)
    trainer.threshold = {'t1': 0, 't2': 0}  # threshold for positive and negative sample

    distributer = create_supervised_evaluator(model, metrics={'dist': DistributionMetric()}, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    Sample(output_transform=lambda x: x[2]).attach(trainer, 'sample_rate')

    # @trainer.on(Events.EPOCH_STARTED)
    # def adjust_learning_rate(engine):
    #     scheduler.step()

    @trainer.on(Events.EPOCH_STARTED)
    def get_train_features(engine):
        model.eval()
        train_feats = []
        train_labels = []
        for batch in train_loader:
            imgs, labels = batch
            imgs = imgs.cuda()
            with torch.no_grad():
                feats = F.normalize(model(imgs))  # normalize for cosine similarity
                train_feats.append(feats)
                train_labels.append(labels)
        train_feats = torch.cat(train_feats, dim=0)  # (500, 2048)
        train_labels = torch.cat(train_labels, dim=0)
        engine.train_feats = train_feats
        engine.train_labels = train_labels

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        writer.add_scalar('loss', engine.state.output[0], engine.state.iteration)

        iter = (engine.state.iteration - 1) % len(online_loader) + 1
        if iter % log_period == 0:
            pos_count, pos_correct, neg_count, neg_correct = engine.state.output[1]
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Positive Sample Rate: {:d}/{:d}, "
                        "Negative Sample Rate: {:d}/{:d}, Positive accuracy: {:d}/{:d}, "
                        "Negative accuracy: {:d}/{:d}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(online_loader),
                                engine.state.metrics['avg_loss'],
                                engine.state.metrics['sample_rate'][0], engine.state.metrics['sample_rate'][1],
                                engine.state.metrics['sample_rate'][2], engine.state.metrics['sample_rate'][3],
                                pos_correct, pos_count, neg_correct, neg_count,
                                optimizer.param_groups[0]['lr']))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        """
        Print using time after each epoch.
        """
        logger.info('Epoch {} done, Threshold: {:.3f}/{:.3f}. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, engine.threshold['t1'], engine.threshold['t2'],
                            timer.value() * timer.step_count, online_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_STARTED)
    def update_validation_distribution(engine):
        """
        Get threshold through validation set.
        """
        if engine.state.epoch % dist_period == 0 or engine.state.epoch == 1:
            logger.info('Generating validation distribution... ')
            distributer.run(val_loader)
            t1, t2, pos_sim, neg_sim = distributer.state.metrics['dist']
            writer.add_histogram('positive sample', pos_sim, engine.state.epoch, bins='auto')
            writer.add_histogram('negative sample', neg_sim, engine.state.epoch, bins='auto')
            # matplotlib
            fig = plt.figure(figsize=(9, 5))
            plt.hist(pos_sim, bins=80, density=True, color='blue', label='positive sample')
            plt.hist(neg_sim, bins=80, density=True, color='red', label='negative sample')
            plt.xlabel('similarity')
            plt.ylabel('frequency')
            plt.title('distribution')
            plt.legend(loc='best')
            writer.add_figure('Distribution', fig, engine.state.epoch)

            # update threshold
            engine.threshold['t1'] = t1
            engine.threshold['t2'] = t2

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(engine):
        """
        Get test results on query/gallery set.
        """
        if engine.state.epoch % eval_period == 0:
            evaluator.run(test_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(online_loader, max_epochs=epochs)
