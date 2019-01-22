# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

sys.path.append('.')
from config import cfg
from data import make_online_loader
from engine.online_trainer import do_train
from modeling import build_online_model
from layers import make_online_loss
from solver import make_optimizer
from utils.logger import setup_logger


def train(cfg):
    # prepare dataset
    train_loader, val_loader, online_loader, test_loader, num_query = make_online_loader(cfg)

    # prepare model
    model = build_online_model(cfg)

    optimizer = make_optimizer(cfg, model)
    # scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
    #                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)

    loss_func = make_online_loss(cfg)

    arguments = {}

    do_train(
        cfg,
        model,
        train_loader,
        online_loader,
        val_loader,
        test_loader,
        optimizer,
        loss_func,
        num_query
    )


def main():
    parser = argparse.ArgumentParser(description="ReID Online Training")
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
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_online", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
