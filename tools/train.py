# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
import numpy as np
from collections import defaultdict

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


def online_train(cfg):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    # model = build_model(cfg, 0)
    # load pretrained model
    # model.load_weight('/export/home/lxy/online-reid/logs/duke2market_baseline/resnet50_model_350.pth')
    # model = End2End_AvgPooling(2, 0.5, 2048, 702)
    # model.load_weight('/export/home/lxy/online-reid/logs/duke2market_paper_model/resnet50_model_350.pth')
    online_model = End2End_AvgPooling(2, 0.5, 2048, 0)
    online_model.load_weight('/export/home/lxy/online-reid/logs/duke2market_paper_model/resnet50_model_350.pth')
    online_model.to('cuda')

    # model.load_weight('./logs/duke2market_paper_model_remove_downsample/resnet50_model_350.pth')
    # optimizer = make_optimizer(cfg, model)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func = make_loss(cfg)

    arguments = {}

    # do_train(
    #     cfg,
    #     model,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     scheduler,
    #     loss_func,
    #     num_query
    # )

    # do_online_train(
    #       0,
    #       cfg,
    #       model,
    #       online_train,
    #       val_loader,
    #       num_query
    # )

    # cluster merge
    # do_online_train(0, cfg, model, train_loader, online_train, loss_func, val_loader, num_query)

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

    # do_online_train(0, cfg, model, train_loader, online_set, loss_func, val_loader, num_query)
    # for on_step in online_dict:
    #     online_set = online_dict[on_step]
    #     # reorganize index
    #     for i, d in enumerate(online_set):
    #         d[3] = i
    #
    #     cluster_model = End2End_AvgPooling(2, 0.5, 2048, 0)
    #     cluster_model.load_weight('/export/home/lxy/online-reid/logs/duke2market_paper_model/resnet50_model_350.pth')
    #     state_dict, new_train_data = do_online_train(on_step, cfg, cluster_model, train_loader,
    #                                                  online_set, loss_func, val_loader, num_query)

        # import torch
        # torch.save(state_dict, '/export/home/lxy/online-reid/logs/online/model_{}.pth'.format(on_step))
        # label_online_set

        # knowledge distillation
        # TODO

        # update online model
        # if on_step == 0:
        #     # initialize online model
        #     online_model.load_state_dict(state_dict)
        # else:
    import torch
    for i in range(8):
        state_dict = torch.load('/export/home/lxy/online-reid/logs/online/model_{}.pth'.format(i))
        for key, value in state_dict.items():
            if key in online_model.state_dict():
                online_param = online_model.state_dict()[key].data
                online_model.state_dict()[key].data.copy_(0.8 * online_param + 0.2 * value.data)

        # test online model performance
        inference(cfg, online_model, val_loader, num_query)


def train(cfg):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    # model = build_model(cfg, 0)
    # load pretrained model
    # model.load_weight('/export/home/lxy/online-reid/logs/duke2market_baseline/resnet50_model_350.pth')
    model = End2End_AvgPooling(1, 0.5, 2048, 702)
    # model = End2End_AvgPooling(0.5, 2048, 0)
    # model.load_weight('./logs/duke2market_paper_model/resnet50_model_350.pth')
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
        num_query
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
    output_dir = os.path.join(os.getcwd() + '/logs', cfg.OUTPUT_DIR)
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
    online_train(cfg)
    # train(cfg)


if __name__ == '__main__':
    main()
