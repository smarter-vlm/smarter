import os
import pdb

import nltk
import numpy as np
import torch

import utils


class BERT:
    # https://huggingface.co/docs/transformers/model_doc/bert
    def __init__(self):
        super(BERT, self).__init__()
        from transformers import BertModel, BertTokenizer

        self.model = BertModel.from_pretrained("bert-base-multilingual-cased").to(
            "cuda"
        )
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        self.word_dim = 768

    def get_word_dim(self):
        return self.word_dim

    def word_embed(self, sentence):
        with torch.no_grad():
            inputs = self.tokenizer(sentence, return_tensors="pt", padding=True).to(
                "cuda"
            )
            outputs = self.model(**inputs)
            word_reprs = outputs.last_hidden_state
        return torch.tensor(word_reprs.squeeze()).cuda()


def globals_init(args):
    global puzzle_diff, puzzle_diff_str, osp, rand, MAX_VAL, MAX_DECODE_STEPS, max_qlen
    global num_puzzles, seed, icon_class_ids, signs
    global SEQ_PUZZLES, NUM_CLASSES_PER_PUZZLE, device, SMART_DATASET_INFO_FILE
    global word_dim, word_embed
    global puzzles_not_included, num_actual_puzz
    global PS_VAL_IDX, PS_TEST_IDX

    device = "cuda"
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

    if args.word_embed == "bert":
        Embed = BERT()
        word_dim = Embed.get_word_dim()
        word_embed = Embed.word_embed
    else:
        print("word embedding used is %s" % (args.word_embed))
