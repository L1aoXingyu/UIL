# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from comet_ml import Experiment
import argparse, os, sys, time
import numpy as np
from collections import defaultdict
import torch
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_online_train
from engine.inference import inference
from modeling import build_model
from modeling.baseline import End2End_AvgPooling
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger


def online_train(cfg, experiment):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    online_model = End2End_AvgPooling(1, 0.5, 2048, 0)
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/logs'
    #                                     '/duke2market_paper_model_remove_downsample/resnet50_model_350.pth'))
    online_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
                                        '/duke2market/resnet50_model_350.pth'))

    # online_model = End2End_AvgPooling(2, 0.5, 2048, 0)
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/logs/duke2market_paper_model'
    #                                     '/resnet50_model_350.pth'))
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/logs'
    #                                     '/duke_paper_model_baseline/resnet50_model_350.pth'))

    online_model.to('cuda')
    # inference(cfg, online_model, val_loader, num_query)
    # model.load_weight(torch.load('./logs/duke2market_paper_model_remove_downsample/resnet50_model_350.pth'))


    arguments = {}

    loss_func = make_loss(cfg)


    # prepare online dataset
    online_dict = defaultdict(list)
    increment_id = 100
    current_id = 100
    chosed_id = 0
    for d in online_train:
        if d[1] < current_id:
            online_dict[chosed_id].append(list(d))
        else:
            current_id += increment_id
            chosed_id += 1

    # ==========
    for on_step in online_dict:
        online_set = online_dict[on_step]
        # reorganize index
        for i, d in enumerate(online_set):
            d[3] = i

        cluster_model = End2End_AvgPooling(1, 0.5, 2048, 0)
        # cluster_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
        #                                     '/duke2market/resnet50_model_350.pth'))

        # cluster_model.load_weight('/export/home/lxy/online-reid/logs/duke2market_paper_model_remove_downsample/resnet50_model_350.pth')
        cluster_model.load_weight(online_model.state_dict())
        with experiment.train():
            state_dict = do_online_train(on_step, cfg, cluster_model, train_loader, online_set, loss_func, val_loader,
                                         num_query, experiment)

        torch.save(state_dict, cfg.OUTPUT_DIR + '/model_{}.pth'.format(on_step))
        # torch.save(state_dict, '/export/home/lxy/online-reid/iccv_logs'
        #                        '/online_iter/model_{}.pth'.format(on_step))

        # state_dict = torch.load('/export/home/lxy/online-reid/logs/online_pretrain/model_{}.pth'.format(on_step))
        # update online model
        for key, value in state_dict.items():
            if key in online_model.state_dict():
                online_param = online_model.state_dict()[key].data
                # if on_step < 6:
                online_model.state_dict()[key].data.copy_(0.9 * online_param + 0.1 * value.data)
                # else:
                    # online_model.state_dict()[key].data.copy_(0.95 * online_param + 0.05 * value.data)

        # test online model performance
        with experiment.test():
            inference(cfg, online_model, val_loader, num_query, experiment)
    # ==========

    # best_perf = 0.54
    # for i in range(8):
    #     state_dict = torch.load('/export/home/lxy/online-reid/logs/online/model_{}.pth'.format(i))
    #     for coef in np.arange(0.5, 0.9, 0.1):
    #         online_state_dict = online_model.state_dict().copy()
    #         for key, value in state_dict.items():
    #             if key in online_state_dict:
    #                 online_param = online_model.state_dict()[key].data
    #                 online_state_dict[key].data.copy_(coef * online_param + (1-coef) * value.data)
    #
    #         online_model.load_state_dict(online_state_dict)
    #         # test online model performance
    #         online_perf = inference(cfg, online_model, val_loader, num_query)

def cross_train(cfg):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    online_model = End2End_AvgPooling(1, 0.5, 2048, 0)
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/logs'
    #                                     '/duke2market_paper_model_remove_downsample/resnet50_model_350.pth'))
    online_model.load_weight(torch.load('/export/home/lxy/online-reid/iccv_logs'
                                        '/duke2market/resnet50_model_350.pth'))

    # online_model = End2End_AvgPooling(2, 0.5, 2048, 0)
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/logs/duke2market_paper_model'
    #                                     '/resnet50_model_350.pth'))
    # online_model.load_weight(torch.load('/export/home/lxy/online-reid/logs'
    #                                     '/duke_paper_model_baseline/resnet50_model_350.pth'))

    online_model.to('cuda')
    # inference(cfg, online_model, val_loader, num_query)
    # model.load_weight(torch.load('./logs/duke2market_paper_model_remove_downsample/resnet50_model_350.pth'))


    arguments = {}

    loss_func = make_loss(cfg)

    # cluster merge
    do_online_train(0, cfg, online_model, train_loader, online_train, loss_func, val_loader, num_query)

def train(cfg, experiment):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = End2End_AvgPooling(1, 0.5, 2048, num_classes)
    optimizer = make_optimizer(cfg, model)
    scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                                  cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func = make_loss(cfg)

    arguments = {}

    do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_func,
        num_query,
        experiment
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    output_dir = os.path.join(os.getcwd() + '/iccv_logs', cfg.OUTPUT_DIR + time.strftime(".%m_%d_%H:%M:%S"))

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cfg.OUTPUT_DIR = output_dir
    cfg.freeze()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.MODEL.CUDA)

    logger = setup_logger("reid_online", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    cudnn.benchmark = True

    experiment = Experiment(api_key='x9ZFXG3YxRLmHdyyDdliXHF5E', project_name='online_reid')
    # experiment = Experiment(api_key='x9ZFXG3YxRLmHdyyDdliXHF5E', project_name='reid_baseline')
    experiment.log_parameters(cfg)

    online_train(cfg, experiment)
    # train(cfg, experiment)
    # cross_train(cfg)


if __name__ == '__main__':
    main()
