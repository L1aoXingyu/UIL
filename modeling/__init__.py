# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .baseline import Baseline, OnlineModel


def build_model(cfg, num_classes):
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH)
    return model


def build_online_model(cfg):
    model = OnlineModel(cfg.MODEL.LAST_STRIDE)
    model.load_weight(cfg.MODEL.BASEMODEL_PATH)
    return model
