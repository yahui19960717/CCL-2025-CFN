#!/usr/bin/python3

import argparse
import os


def construct_hyper_param():
    parser = argparse.ArgumentParser()
    # train settings
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument("--batch_size", default=9, type=int, # 12
                        help="Batch size") # 3
    parser.add_argument("--update_freq", default=1, type=int,
                        help="update_freq")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--lr', default=2e-5, type=float, help='model learning rate.')

    # bert settings
    parser.add_argument("--config_file",
                        # default='./chinese_bert_wwm_ext/config.json',
                        default='./chinese_robert_wwm_ext_large/config.json',
                        type=str,
                        help="bert_config_file")
    parser.add_argument("--vocab_file",
                        # default='./chinese_bert_wwm_ext/vocab.txt',
                        default='./chinese_robert_wwm_ext_large/vocab.txt',
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--init_checkpoint",
                        # default='./chinese_bert_wwm_ext/pytorch_model.bin',
                        default='./chinese_robert_wwm_ext_large/pytorch_model.bin',
                        type=str,
                        help="bert_init_checkpoint")

    parser.add_argument("--with_adv_train",
                        action='store_true',
                        help="with_adv_train")

    parser.add_argument("--data_dir",
                        default='./data', type=str,
                        help="save_dir")

    parser.add_argument("--save_dir",
                        default='./models', type=str,
                        help="save_dir")

    parser.add_argument("--focal_loss",
                        action='store_true',
                        help="focal_loss")
    parser.add_argument("--gamma",
                        default=1.0,
                        type=float,
                        help="gamma")
    parser.add_argument("--use_pos",
                        action='store_true',
                        help="use_pos")
    parser.add_argument("--use_dep",
                        action='store_true',
                        help="use_dep") 
    parser.add_argument("--use_ws",
                        action='store_true',
                        help="use_ws")   

    parser.add_argument("--embed_file",
                        # default='./chinese_bert_wwm_ext/vocab.txt',
                        default='./dataset/cc.zh.300.cfnAB.vec',
                        type=str,
                        help="The embedding file from fasttext")        
    parser.add_argument("--use_embedding",
                        action='store_true',
                        help="use_embedding")
    args = parser.parse_args()

    return args



args = construct_hyper_param()

