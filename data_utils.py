# Minimally edited from https://github.com/merlresearch/SMART
import os
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
import pdb
import pickle

import nltk
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import text_encoder as gv
import utils


class SMART_Data(Dataset):
    def __init__(self, args):
        vocab_path = args.vocab_path
        self.max_qlen = 110
        self.max_olen = 4  # max option length
        self.use_word_embed = False
        self.word_embed = None
        self.im_side = 224
        self.preprocess = args.preprocess

        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)
            print("vocabulary size = %d" % (len(self.vocab)))

        if args.preprocess is None:  # VL models, will do preprocess later.
            self.transform = Compose(
                [
                    Resize(
                        224
                    ),  # if the images are of higher resolution. we work with pre-resized 224x224 images.
                    ToTensor(),
                    Normalize(torch.Tensor([0.5]), torch.Tensor([0.5])),
                ]
            )
        elif args.model_name in [
            "dinov2",
            "siglip",
            "fused_dinov2_siglip",
        ]:  # this will do feature extractin later.
            self.transform = Compose(
                [
                    Resize(224),
                    ToTensor(),
                ]
            )
        else:
            self.transform = args.preprocess

    def apply_transform(self, im_path):
        im = Image.open(im_path).convert("RGB")
        return self.transform(im)

    def quest_encode(self, question):
        tokens = nltk.tokenize.word_tokenize(question.lower())
        q_enc = np.zeros((self.max_qlen,), dtype="long")
        enc_tokens = (
            [self.vocab("<start>")]
            + [self.vocab(tokens[t]) for t in range(len(tokens))]
            + [self.vocab("<end>")]
        )
        q_enc[: min(self.max_qlen, len(enc_tokens))] = np.array(enc_tokens)
        return q_enc

    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def opts_encode(self, opts, key):
        opts = opts.lower()
        tokens = nltk.tokenize.word_tokenize(opts)
        enc_tokens = [self.vocab(tokens[t]) for t in range(len(tokens))]
        opt_enc = np.zeros((self.max_olen,), dtype="long")
        opt_enc[: min(self.max_olen, len(enc_tokens))] = np.array(enc_tokens)
        return opt_enc

    def split_puzzles(self, puzzle_ids, split_ratio, split_name, split_type="standard"):

        splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
        n = len(puzzle_ids)
        if split_name == "train":
            st = 0
            en = int(np.floor(n * splits[0] / 100.0))
            puzzle_ids = puzzle_ids[st:en]
        elif split_name == "val":
            st = int(np.ceil(n * splits[0] / 100.0))
            en = int(np.floor(n * splits[1] / 100.0))
            puzzle_ids = puzzle_ids[st:en]
        else:
            st = int(np.ceil(n * splits[1] / 100.0))
            puzzle_ids = puzzle_ids[st:]
        print("puzzles for %s =" % (split_name))
        print(puzzle_ids)
        return puzzle_ids

    def split_data(self, info, split_ratio, split_name, split_type="standard"):
        """
        split_type=standard is to use the split_ratio in the instance order
        split_type=exclude is to exclude answers from the split, e.g., train on all answers except say 1, and test 1
        split_type=puzzle is to split the puzzles into the respective ratios. so we don't have to do anything here.
        """
        if split_type == "standard":
            splits = np.array([int(spl) for spl in split_ratio.split(":")]).cumsum()
            n = len(info)
            if split_name == "train":
                st = 0
                en = int(np.floor(n * splits[0] / 100.0))
                info = info[st:en]
            elif split_name == "val":
                st = int(np.ceil(n * splits[0] / 100.0))
                en = int(np.floor(n * splits[1] / 100.0))
                info = info[st:en]
            else:
                st = int(np.ceil(n * splits[1] / 100.0))
                info = info[st:]

        else:
            raise "Unknown puzzle split type!!"

        return info


class SMART_TrainData(SMART_Data):
    def __init__(self, args, split):
        super().__init__(args)
        self.data_root = args.data_root
        self.num_tot = args.data_tot  # how many instances per puzzles should we use?
        self.diff = args.train_diff
        self.word_embed = args.word_embed
        self.qa_info = []
        train_pids = None

        puzzle_ids = args.puzzle_ids
        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])

            qa_info = utils.read_csv(
                os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id
            )
            qa_info = qa_info[: self.num_tot]

            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(
                    qa_info[t], qa_info[t]["Answer"]
                )
            self.qa_info = self.qa_info + self.split_data(
                qa_info, args.split_ratio, split, "standard"
            )

            gv.MAX_VAL = max(gv.MAX_VAL, gv.NUM_CLASSES_PER_PUZZLE[puzzle_id])
        print("num_train=%d max_answer_value=%d" % (len(self.qa_info), gv.MAX_VAL))
        print("split=%s puzzle_ids=" % (split), end=" ")
        print(puzzle_ids)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = pid + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im = self.apply_transform(
            gv.osp(self.data_root, puzzle_root, "img", info["image"])
        )
        qa = self.quest_encode(info["Question"])
        opts = 0
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(
            gv.MAX_DECODE_STEPS,
        )
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            try:
                answer[: len(answer_value)] = answer_value
            except:
                print(info)
                pdb.set_trace()
        return (
            im,
            torch.tensor(qa),
            opts,
            torch.tensor(lbl),
            torch.tensor(answer),
            torch.tensor(int(pid)),
        )

    def __len__(self):
        return len(self.qa_info)


class SMART_ValData(SMART_Data):
    def __init__(self, args, split):
        super().__init__(args)
        self.data_root = args.data_root
        self.num_tot = args.data_tot
        self.word_embed = args.word_embed
        self.qa_info = []

        self.diff = args.test_diff if split == "test" else args.train_diff
        puzzle_ids = args.puzzle_ids

        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(
                os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id
            )
            qa_info = qa_info[: self.num_tot]
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(
                    qa_info[t], qa_info[t]["Answer"]
                )
            self.qa_info = self.qa_info + self.split_data(
                qa_info, args.split_ratio, split, "standard"
            )
            gv.MAX_VAL = max(gv.MAX_VAL, gv.NUM_CLASSES_PER_PUZZLE[puzzle_id])
        print("num_val = %d max_answer_value=%d" % (len(self.qa_info), gv.MAX_VAL))
        print("split=%s puzzle_ids=" % (split), end=" ")
        print(puzzle_ids)

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = info["puzzle_id"] + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im = self.apply_transform(
            gv.osp(self.data_root, puzzle_root, "img", info["image"])
        )
        qa = self.quest_encode(info["Question"])

        _ = [utils.str_replace_(info, key) for key in ["A", "B", "C", "D", "E"]]
        opts = [
            utils.get_val(info, key, is_one_of_option=True)
            for key in ["A", "B", "C", "D", "E"]
        ]
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(
            gv.MAX_DECODE_STEPS,
        )
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            answer[: len(answer_value)] = answer_value

        return (
            im,
            torch.tensor(qa),
            opts,
            torch.tensor(lbl),
            torch.tensor(answer),
            torch.tensor(int(info["puzzle_id"])),
        )

    def __len__(self):
        return len(self.qa_info)


def SMART_collate_fn(data):
    """we use it only for val and test to load the options as a list"""
    concat = lambda data_list: torch.cat([x.unsqueeze(0) for x in data_list])
    im, qa, opts, lbl, answer, puzzle_ids = zip(*data)
    im = concat(im).float()
    qa = concat(qa)
    lbl = concat(lbl)
    answer = concat(answer)
    puzzle_ids = concat(puzzle_ids)
    # print("what is opts in collate", opts)
    return im, qa, opts, lbl, answer, puzzle_ids
