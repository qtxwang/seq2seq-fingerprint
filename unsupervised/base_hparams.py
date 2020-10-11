""" Hyper parameters"""
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

def build_base_hparams():
    """build hyper-parameters"""
    hparams = [
            hp.HParam("dropout_rate", hp.Discrete([0.5])),
            hp.HParam("num_layers", hp.Discrete([3])),
            hp.HParam("size",hp.Discrete([128])),
            hp.HParam("learning_rate", hp.Discrete([0.5])),
            hp.HParam("learning_rate_decay_factor", hp.Discrete([0.99])),
            hp.HParam("target_vocab_size", hp.Discrete([41])),
            hp.HParam("batch_size", hp.Discrete([256])),
            hp.HParam("source_vocab_size", hp.Discrete([41])),
            hp.HParam("max_gradient_norm", hp.Discrete([5.0]))
        ]
    return hparams
