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
from pathlib import Path

import comet_ml
import pytorch_lightning as pl

import numpy as np
from comet_ml import Experiment
from comet_ml.integration.pytorch import log_model

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "1"

import warnings

warnings.filterwarnings("ignore")
import argparse
import copy
import time

import torch.nn.functional as F
from tqdm import tqdm

import vocab_utils
import data_utils as dl
import text_encoder as gv
import losses
import deep_vlm_reasoners
import utils

from transformers.optimization import get_cosine_schedule_with_warmup

AVAIL_GPUS = min(1, torch.cuda.device_count())
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Available GPUs {AVAIL_GPUS} and current device {device}")

API_KEY = Path("modules/denisa_vlm_reasoners/.comet_token").read_text().strip()
workspace = Path("modules/denisa_vlm_reasoners/.comet_workspace").read_text().strip()

experiment = Experiment(
    api_key=API_KEY,
    project_name="multimodalai",
    workspace=workspace,
    auto_metric_logging=True,  # default
)

# For training direct baselines from SMART CVPR'23 article: https://github.com/D-Roberts/SMART


def reset_state(args):
    #    global seed
    gv.seed = np.random.randint(10000) if args.seed == -1 else args.seed
    args.seed = gv.seed
    manualSeed = gv.seed  #
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.deterministic = True
    print("seed = %d" % (gv.seed))


def train(args, dataloader, im_backbone):
    criterion = losses.Criterion(args)

    model = deep_vlm_reasoners.Puzzle_Net(args, im_backbone=im_backbone, device=device)

    print(
        f"\n Number trainable params {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    model.to(device)
    # print("\n Model architecture: \n", model)

    log_model(experiment, model, model_name="Puzzle_Net")

    def normalize(err, pids):
        """this function divides the error by the gt number of classes for each puzzle."""
        pids = np.array(pids)
        for t in range(len(err)):
            err[t] = err[t] / gv.NUM_CLASSES_PER_PUZZLE[str(pids[t])]
        return err

    def get_result(out):
        pred_max = F.softmax(out, dim=1).argmax(dim=1).cpu()

        return pred_max

    def save_model(args, net, acc, epoch, location):
        state = {
            "net": net.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }

        if not os.path.isdir(location):
            os.mkdir(location)

        loc = os.path.join(
            location,
            "ckpt_%s_%s_%s.pth" % (args.model_name, args.word_embed, args.seed),
        )
        print("saving checkpoint at %s" % (loc))
        torch.save(state, loc)

    def train_loop(epoch, train_loader, optimizer):
        model.train()
        tot_loss = 0.0

        for i, (im, q, _, a, av, pids) in tqdm(enumerate(train_loader)):

            im = im.float().to(device)
            q = q.to(device)
            a = a.to(device)
            av = av.to(device)

            # the model is puzzlenet
            out = model(im, q, puzzle_ids=pids)
            loss = criterion(out, av, pids)
            loss.backward()

            if not args.run_baseline:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            optimizer.step()
            if not args.run_baseline:
                scheduler.step()
            optimizer.zero_grad()

            tot_loss += loss.item()

            experiment.log_metrics({"train_batch_loss": loss.item()}, step=i)

        tot_loss /= float(i)
        return tot_loss

    def val_loop(val_loader, model):
        model.eval()
        acc_mean = 0
        val_tot_loss = 0.0
        cnt = 0
        err_mean = 0
        puzzle_acc = {}
        with torch.no_grad():
            for i, (im, q, _, a, av, pids) in enumerate(val_loader):

                q = q.to(device)
                im = im.float()
                im = im.to(device)
                av = av.to(device)

                out = model(im, q, puzzle_ids=pids)
                val_loss = criterion(out, av, pids)
                val_tot_loss += val_loss.item()

                experiment.log_metrics({"val_batch_loss": val_loss.item()}, step=i)

                av = av.cpu()
                upids = torch.unique(pids)
                acc = 0
                error = 0
                for t in upids:
                    idx = pids == t
                    tt = t.item()

                    if t not in gv.SEQ_PUZZLES:
                        pred_max = get_result(out[int(tt)])
                        pacc = (pred_max == av[idx, 0]).sum()
                        perror = normalize(np.abs(pred_max - av[idx, 0]), pids).sum()
                    else:
                        pred_ans = []
                        pacc = 1
                        for k in range(gv.MAX_DECODE_STEPS):
                            pred_max = get_result(out[int(tt)][k])
                            pred_ans.append(pred_max)
                            pacc = pacc * (pred_max == av[idx][:, k])
                        pacc = pacc.sum()
                        perror = 0

                    if str(tt) in puzzle_acc.keys():
                        puzzle_acc[str(tt)][0] += pacc
                        puzzle_acc[str(tt)][2] += idx.sum()
                    else:
                        puzzle_acc[str(tt)] = [pacc, 0, idx.sum()]

                    acc += pacc
                    error += perror

                acc_mean += acc
                err_mean += error
                cnt += len(av)

        return (
            acc_mean / float(cnt),
            err_mean / float(cnt),
            puzzle_acc,
            val_tot_loss / len(val_loader),
        )

    def test_loop(test_loader, model):

        acc, err, puzzle_acc, test_ep_loss = val_loop(test_loader, model)
        class_perf = utils.print_puzz_acc(args, puzzle_acc, log=True)
        print(
            "***** Final Test Performance: S_acc = %0.2f Prediction Variance = %0.2f "
            % (acc * 100, err)
        )
        print(f"Test class perf {class_perf}")
        print(f"Test val loss: val_ep_loss, {test_ep_loss}")

    if args.test:
        deep_vlm_reasoners.load_pretrained_models(args, args.model_name, model=model)
        test_loop(dataloader["test"], model)
        return

    if not args.run_baseline:

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, args.beta2),
            eps=args.eps,
            weight_decay=args.wd,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    train_loader = dataloader["train"]
    val_loader = dataloader["valid"]
    test_loader = dataloader["test"]

    num_steps = args.num_epochs * len(train_loader)

    num_warmup_steps = 10  # TODO DR: no hardcoding

    if not args.run_baseline:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_steps
        )

    # training loop
    best_model = None
    best_acc = 0
    no_improvement = 0
    num_thresh_epochs = 3

    # stop training if there is no improvement after this.
    print("starting training...")
    for epoch in range(args.num_epochs):
        tt = time.time()
        model.train()

        # jsut in case
        optimizer.zero_grad()

        loss = train_loop(epoch, train_loader, optimizer)

        experiment.log_metrics({"epoch_train_loss": loss}, epoch=epoch)

        tt = time.time() - tt

        if epoch >= 0:  # always eval
            model.eval()

            acc, err, puz_acc, val_tot_loss = val_loop(val_loader, model)
            experiment.log_metrics(
                {
                    "val_acc": acc,
                    "val_var": err,
                    # "val_oacc": oacc,
                    "val_epoch_loss": val_tot_loss,
                },
                epoch=epoch,
            )

            class_avg_perf = utils.print_puzz_acc(args, puz_acc, log=args.log)

            with experiment.context_manager("val_acc"):
                experiment.log_metrics(
                    {k: v[0] for k, v in class_avg_perf.items()}, epoch=epoch
                )

            if acc >= best_acc:
                best_epoch = epoch
                best_acc = acc
                best_model = copy.deepcopy(model)
                save_model(args, best_model, acc, epoch, args.location)
                no_improvement = 0
            else:
                no_improvement += 1
                if no_improvement > num_thresh_epochs:
                    print("no training improvement... stopping the training.")
                    class_avg_perf = utils.print_puzz_acc(args, puz_acc, log=args.log)
                    break

            print(
                "%d) Time taken=%f Epoch=%d Train_loss = %f S_acc = %f Variance = %f Best S_acc (epoch) = %f (%d)\n"
                % (
                    gv.seed,
                    tt,
                    epoch,
                    loss,
                    acc * 100,
                    err,
                    best_acc * 100,
                    best_epoch,
                )
            )

    test_loop(test_loader, best_model)


def get_data_loader(
    args, split, batch_size=100, shuffle=True, num_workers=6, pin_memory=True
):
    if split == "train":
        dataset = dl.SMART_TrainData(args, split)
        collate_fn = None
    else:
        dataset = dl.SMART_ValData(args, split)
        collate_fn = dl.SMART_collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    return data_loader


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SMART puzzles")
    parser.add_argument(
        "--puzzles",
        default="all",
        type=str,
        help="comma separated / all / puzzle groups (counting,math etc.)",
    )
    parser.add_argument("--batch_size", default=128, type=int, help="batch size (128)")
    parser.add_argument("--num_epochs", default=3, type=int, help="epoch")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate (0.001)")

    parser.add_argument(
        "--data_root",
        type=str,
        default="",
        help="location of the csv files, and location of the images, relative location is provided in the csv file.",
    )
    parser.add_argument(
        "--train_diff", type=str, default="easy", help="easy/medium/hard"
    )
    parser.add_argument(
        "--test_diff", type=str, default="easy", help="easy/medium/hard"
    )
    parser.add_argument(
        "--split_ratio",
        type=str,
        default="60:20:20",
        help="how to split train and val, when both use the same instance list.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./data/v2/",
        help="location to save intermediate files.",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default="none",
        help="location to save intermediate files.",
    )
    parser.add_argument("--num_workers", type=int, default=16, help="number of workers")
    parser.add_argument("--pretrained", type=str, help="should use a pretrained model?")

    parser.add_argument(
        "--model_name",
        type=str,
        help="model to use dinov2/siglip/fused_dinov2_siglip/resnet50",
        default="fused_dinov2_siglip",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed to use")
    parser.add_argument(
        "--data_tot",
        type=int,
        default=1000,
        help="how many instances to use for train+val+test",
    )

    parser.add_argument(
        "--log", action="store_true", help="should print detailed log of accuracy?"
    )

    parser.add_argument(
        "--word_embed", type=str, default="siglip", help="siglip/mbert/bert"
    )

    parser.add_argument(
        "--use_single_image_head",
        action="store_true",
        help="use a single image head for all the puzzles?",
    )
    parser.add_argument(
        "--qf_layer",
        action="store_true",
        help="add a q-former inspired layer to get a composite vision-language representation",
    )

    # Ref for baseline choices https://github.com/D-Roberts/SMART/tree/main (code forked from original paper's repo)
    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="The best performing baseline from the original SMART CVPR article. Select bert and resnet50 for backbones with this.",
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="number attention heads in QFlayer self and cross attention?",
    )

    parser.add_argument("--log_freq", type=int, default=1, help="log frequency?")
    parser.add_argument("--test", action="store_true", help="evaluate a model?")

    parser.add_argument(
        "--repr_size",
        type=int,
        default=128,
        help="intermediate representation size for image and language encoders?",
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.2,
        help="weight decay for AdamW?",
    )

    parser.add_argument(
        "--pdrop",
        type=float,
        default=0.2,
        help="dropout prob?",
    )
    parser.add_argument(
        "--ln_eps",
        type=float,
        default=1e-6,
        help="layernorm eps?",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="adamw eps?",
    )

    parser.add_argument(
        "--beta2",
        type=float,
        default=0.98,
        help="adamw beta2?",
    )
    parser.add_argument(
        "--h_sz",
        type=int,
        default=256,
        help="hidden size in qf intermediate and gru?",
    )

    args = parser.parse_args()

    if args.test:
        assert (
            args.seed > -1
        )  # when evaluating we need to use the seed to take the checkpoint.

    gv.globals_init(args)

    args.puzzle_ids_str, args.puzzle_ids = utils.get_puzzle_ids(args)
    args.location = os.path.join(args.save_root, "checkpoints")
    args.log_path = os.path.join(args.save_root, "log")

    reset_state(args)
    gv.NUM_CLASSES_PER_PUZZLE = utils.get_puzzle_class_info(
        args
    )  # initialize the global with the number of outputs for each puzzle.

    vocab = vocab_utils.process_text_for_puzzle(args)
    if args.vocab_path == "none":
        args.vocab_path = os.path.join(
            args.save_root, "vocab_puzzle_" + args.puzzle_ids_str + ".pkl"
        )

    im_backbone, preprocess = deep_vlm_reasoners.load_pretrained_models(
        args, args.model_name, model=None
    )

    args.preprocess = preprocess

    train_loader = get_data_loader(
        args,
        "train",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = get_data_loader(
        args,
        "val",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = get_data_loader(
        args,
        "test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    dataloader = {
        "train": train_loader,
        "valid": val_loader,
        "test": test_loader,
    }

    utils.backup_code_and_start_logger(args, args.log_path, args.seed)

    print(args)
    print(
        f"\n Num batches: train {len(train_loader)}, val {len(val_loader)}, and test {len(test_loader)}"
    )

    print("num_puzzles=%d" % (len(args.puzzle_ids)))

    train(args, dataloader, im_backbone)
