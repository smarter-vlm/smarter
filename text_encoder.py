"""
Copyright Denisa Roberts 2024

# References
# https://github.com/merlresearch/SMART
# CVPR SMART article https://arxiv.org/pdf/2212.09993.pdf

# adsformers https://ui.adsabs.harvard.edu/abs/2023arXiv230201255A/abstract
# eficient vit image representations https://www.researchgate.net/profile/Denisa-Roberts/publication/370980888_Efficient_Large-Scale_Vision_Representation_Learning/links/64ecf9d99b1e56033da9d827/Efficient-Large-Scale-Vision-Representation-Learning.pdf

# prismatic vlm https://arxiv.org/pdf/2402.07865.pdf
# qformer https://arxiv.org/pdf/2301.12597
# mbert https://link.springer.com/chapter/10.1007/978-3-030-72240-1_36

# siglip https://huggingface.co/google/siglip-so400m-patch14-384
# dinov2 https://huggingface.co/facebook/dinov2-base
"""

import os
import pdb

import nltk
import numpy as np
import torch

import utils
from main_reasoner import device


class mBERT:
    # https://huggingface.co/docs/transformers/model_doc/bert
    def __init__(self):
        super(mBERT, self).__init__()
        from transformers import BertModel, BertTokenizer

        self.model = BertModel.from_pretrained("bert-base-multilingual-cased").to(
            device
        )
        print(
            f"\n Number trainable params before explicit freezing of text backb  {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
        for param in self.model.parameters():

            param.requires_grad = False

        print(
            f"\n Number trainable params after explicit freezing of text backb  {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.word_dim = 768

    def get_word_dim(self):
        return self.word_dim

    def word_embed(self, sentence):
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(
                device
            )
            outputs = self.model(**inputs)
            word_reprs = outputs.last_hidden_state
        return torch.tensor(word_reprs.squeeze()).to(device)


class BERT:
    # https://huggingface.co/docs/transformers/model_doc/bert
    def __init__(self):
        super(BERT, self).__init__()
        from transformers import BertModel, BertTokenizer

        self.model = BertModel.from_pretrained("bert-base-uncased").to(device)
        print(
            f"\n Number trainable params before explicit freezing of text backb  {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
        for param in self.model.parameters():

            param.requires_grad = False

        print(
            f"\n Number trainable params after explicit freezing of text backb  {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.word_dim = 768

    def get_word_dim(self):
        return self.word_dim

    def word_embed(self, sentence):
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(
                device
            )
            outputs = self.model(**inputs)
            word_reprs = outputs.last_hidden_state
        return torch.tensor(word_reprs.squeeze()).to(device)


class Siglip:

    def __init__(self):
        super(Siglip, self).__init__()
        from transformers import SiglipTextModel, AutoTokenizer

        self.model = SiglipTextModel.from_pretrained(
            "google/siglip-base-patch16-224"
        ).to(device)

        print(
            f"\n Number trainable params before explicit freezing of text backb  {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
        for param in self.model.parameters():

            param.requires_grad = False

        print(
            f"\n Number trainable params after explicit freezing of text backb  {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer.model_max_length = 64
        self.word_dim = 768

    def get_word_dim(self):
        return self.word_dim

    def word_embed(self, sentence):
        with torch.no_grad():
            inputs = self.tokenizer(
                sentence, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            outputs = self.model(**inputs)
            word_reprs = outputs.last_hidden_state.mean(1)
        return torch.tensor(word_reprs.squeeze()).to(device)


def globals_init(args):
    global puzzle_diff, puzzle_diff_str, osp, rand, MAX_VAL, MAX_DECODE_STEPS, max_qlen
    global num_puzzles, seed, icon_class_ids, signs
    global SEQ_PUZZLES, NUM_CLASSES_PER_PUZZLE, device, SMART_DATASET_INFO_FILE
    global word_dim, word_embed
    global puzzles_not_included, num_actual_puzz
    global PS_VAL_IDX, PS_TEST_IDX

    # device = "cuda"
    puzzle_diff = {"easy": ""}  # {'easy': 'e', 'medium': 'm', 'hard': 'h'}
    puzzle_diff_str = {"easy": ""}
    osp = os.path.join
    rand = lambda: np.random.rand() > 0.5
    MAX_VAL = 0
    MAX_DECODE_STEPS = 10  # number of steps to decode the LSTM.
    num_puzzles = 101
    max_qlen = 110
    seed = 10
    icon_dataset_path = "./dataset/icon-classes.txt"
    icon_class_ids = utils.get_icon_dataset_classes(icon_dataset_path)
    signs = np.array(["+", "-", "x", "/"])  # puzzle 58
    NUM_CLASSES_PER_PUZZLE = {}
    SEQ_PUZZLES = [16, 18, 35, 39, 63, 100]
    SMART_DATASET_INFO_FILE = "./dataset/SMART_info_v2.csv"
    num_actual_puzz = 102
    puzzles_not_included = set([])
    PS_VAL_IDX = [7, 43, 64]
    PS_TEST_IDX = [
        94,
        95,
        96,
        97,
        98,
        99,
        101,
        61,
        62,
        65,
        66,
        67,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
    ]

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    if args.word_embed == "mbert":
        Embed = mBERT()
        word_dim = Embed.get_word_dim()
        word_embed = Embed.word_embed
    elif args.word_embed == "siglip":
        Embed = Siglip()
        word_dim = Embed.get_word_dim()
        word_embed = Embed.word_embed
    elif args.word_embed == "bert":
        Embed = BERT()
        word_dim = Embed.get_word_dim()
        word_embed = Embed.word_embed
    else:
        print("word embedding used is %s" % (args.word_embed))
