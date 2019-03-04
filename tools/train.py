# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train, do_online_train
from modeling import build_model
from modeling.baseline import End2End_AvgPooling
from layers import make_loss
from solver import make_optimizer, WarmupMultiStepLR

from utils.logger import setup_logger


def train(cfg):
    # prepare dataset
    train_loader, online_train, val_loader, num_query, num_classes = make_data_loader(cfg)

    # prepare model
    model = build_model(cfg, 0)
    # load pretrained model
    # model.load_weight('/export/home/lxy/online-reid/logs/duke2market_baseline/resnet50_model_350.pth')
    # model = End2End_AvgPooling(0.5, 2048)

    # optimizer = make_optimizer(cfg, model)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    # loss_func = make_loss(cfg)

    arguments = {}

    # do_train(
    #     cfg,
    #     model,
    #     train_loader,
    #     online_train,
    #     val_loader,
    #     optimizer,
    #     scheduler,
    #     loss_func,
    #     num_query
    # )

    do_online_train(
        cfg,
        model,
        online_train,
        val_loader,
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
    train(cfg)


if __name__ == '__main__':
    main()
