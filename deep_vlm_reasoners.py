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
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
import pdb
import pickle

import numpy as np
import torch.nn.functional as F
from PIL import Image


import text_encoder as gv
from layers import (
    QFLayer,
    CLayer,
    QV_Fusion,
    PuzzleMLPDecoder,
    get_activation_fn,
    get_activation_layer,
)


class Puzzle_Net(nn.Module):
    def __init__(self, args, im_backbone, device):
        super(Puzzle_Net, self).__init__()
        vocab_path = args.vocab_path
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.args = args
        self.device = device

        self.num_opts = 5
        self.out_dim = args.repr_size
        self.h_sz = args.h_sz
        self.model_name = args.model_name
        self.use_single_image_head = args.use_single_image_head
        self.word_embed = args.word_embed
        self.sorted_puzzle_ids = np.sort(np.array([int(ii) for ii in args.puzzle_ids]))

        self.max_val = gv.MAX_VAL + 1

        # Image backbones - frozen
        if args.model_name[:6] == "resnet":
            self.im_repr_size = im_backbone.fc.weight.shape[1]
            modules = list(im_backbone.children())[:-1]
            self.im_cnn = nn.Sequential(*modules)

        elif args.model_name in ["dinov2"]:
            self.preprocess = args.preprocess
            self.im_cnn = lambda x: self.process_dinov2(x)
            self.im_backbone = im_backbone
            self.im_repr_size = 768

        elif args.model_name in ["siglip"]:
            self.preprocess = args.preprocess
            self.im_cnn = lambda x: self.process_dinov2(x)
            self.im_backbone = im_backbone
            self.im_repr_size = 768

        # Reference adsformers and prismatic
        elif args.model_name in ["fused_dinov2_siglip"]:
            from transformers import AutoImageProcessor

            image_processor_siglip = AutoImageProcessor.from_pretrained(
                "google/siglip-base-patch16-224"
            )
            image_processor_dino = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-base"
            )
            self.preprocess = None
            self.im_cnn = lambda x: self.process_fused_vision(
                x, image_processor_siglip, image_processor_dino
            )
            self.im_backbone = im_backbone
            self.im_repr_size = 768 + 768

        else:
            raise "unknown model_name %s" % (args.model_name)

        self.create_puzzle_head(args)

        # Language backbones - frozen
        if args.word_embed in ["siglip"]:
            self.siglip_dim = 768
            self.q_MLP = nn.Sequential(
                nn.Linear(self.siglip_dim, self.h_sz),
                get_activation_layer(args.run_baseline),
                nn.Linear(self.h_sz, self.out_dim),
            )
        else:
            # bert and mbert
            word_dim = gv.word_dim
            self.q_emb = nn.Identity()
            self.q_lstm = nn.GRU(
                int(word_dim),
                int(self.h_sz),
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                bias=args.run_baseline,
            )
            self.q_MLP = nn.Linear(self.h_sz * 2, self.out_dim)

        self.o_encoder = nn.Sequential(
            nn.Embedding(len(self.vocab), self.out_dim, max_norm=1),
            nn.Linear(self.out_dim, self.out_dim),
            get_activation_layer(args.run_baseline),
        )

        if args.qf_layer:
            composite_dim = 2 * 768 + self.args.repr_size
            self.qv_fusion = QV_Fusion(
                composite_dim, self.out_dim, args=self.args
            )  # 1664
            self.c = CLayer(dim=composite_dim, args=self.args)

        else:
            if not args.run_baseline:
                self.qv_fusion = QV_Fusion(
                    2 * self.out_dim, self.out_dim, args=self.args
                )
                self.c = CLayer(dim=2 * self.out_dim, args=self.args)
            else:
                self.qv_fusion = nn.Sequential(
                    nn.Linear(self.out_dim * 2, self.out_dim),
                    nn.ReLU(),
                    nn.Linear(self.out_dim, self.out_dim),
                    nn.ReLU(),
                )

        if args.qf_layer:
            self.qf = QFLayer(num_heads=args.num_heads, args=self.args)

        self.create_puzzle_tail(args)

    def process_dinov2(self, x):
        x = self.decode_image(x)
        inputs = self.preprocess(images=x, do_rescale=True, return_tensors="pt").to(
            self.device
        )
        outputs = self.im_backbone(**inputs)
        return outputs.last_hidden_state.mean(1)

    def process_fused_vision(self, x, image_processor_siglip, image_processor_dino):
        x = self.decode_image(x)

        inputs_din = image_processor_dino(
            images=x, do_rescale=True, return_tensors="pt"
        ).to(self.device)
        inputs_sig = image_processor_siglip(
            images=x, do_rescale=True, return_tensors="pt"
        ).to(self.device)

        im_backbone_din, im_backbone_sig = self.im_backbone

        im_backbone_din = im_backbone_din.to(self.device)
        im_backbone_sig = im_backbone_sig.to(self.device)

        outputs_din = im_backbone_din(**inputs_din)
        outputs_sig = im_backbone_sig(**inputs_sig)

        return torch.cat(
            [
                outputs_din.last_hidden_state.mean(1),
                outputs_sig.last_hidden_state.mean(1),
            ],
            dim=1,
        )

    def create_puzzle_head(self, args):
        if args.use_single_image_head:
            self.im_encoder = nn.Sequential(
                nn.Linear(self.im_repr_size, self.out_dim),
                get_activation_layer(args.run_baseline),
                nn.Linear(self.out_dim, self.out_dim),
            )
        else:
            self.puzzle_ids = args.puzzle_ids
            im_encoder = [nn.Sequential(nn.Linear(self.out_dim, 1))]
            for i in range(1, gv.num_puzzles + 1):
                im_encoder.append(
                    nn.Sequential(
                        nn.Linear(self.im_repr_size, self.out_dim),
                        get_activation_layer(args.run_baseline),
                        nn.Linear(self.out_dim, self.out_dim),
                    )
                )
            self.im_encoder = nn.ModuleList(im_encoder)

    def create_puzzle_tail(self, args):
        self.puzzle_ids = args.puzzle_ids
        ans_decoder = [
            nn.Sequential(nn.Linear(self.out_dim, 1))
        ]  # start with a dummy as we are 1-indexed wrt puzzle ids.
        if args.puzzles == "all":
            puzzles = range(1, gv.num_puzzles + 1)
        else:
            puzzles = self.puzzle_ids
        for pid in puzzles:
            num_classes = gv.NUM_CLASSES_PER_PUZZLE[str(pid)]
            if int(pid) not in gv.SEQ_PUZZLES:
                if not args.run_baseline:
                    dec = PuzzleMLPDecoder(self.out_dim, num_classes)
                    ans_decoder.append(dec)
                else:
                    ans_decoder.append(
                        nn.Sequential(
                            nn.Linear(self.out_dim, self.out_dim),
                            nn.ReLU(),
                            nn.Linear(self.out_dim, self.out_dim),
                            nn.ReLU(),
                            nn.Linear(self.out_dim, num_classes),
                        )
                    )
            else:
                if args.run_baseline:
                    ans_decoder.append(
                        nn.LSTM(
                            int(self.out_dim),
                            int(num_classes),
                            num_layers=1,
                            batch_first=True,
                        )
                    )

                else:
                    ans_decoder.append(
                        nn.GRU(
                            int(self.out_dim),
                            int(num_classes),
                            num_layers=1,
                            batch_first=True,
                        )
                    )
        self.ans_decoder = nn.ModuleList(ans_decoder)

    def decode_image(self, im_list):
        """convert torch tensor images back to Image."""
        #        im_list = (im_list +1)/2. # this is in range [0, 1].
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [
            Image.fromarray(im_list[ii]) for ii in range(len(im_list))
        ]  # convert im
        return im_list

    def save_grad_hook(self):
        self.vis_grad = None

        def bwd_hook(module, in_grad, out_grad):
            self.vis_grad = out_grad

        return bwd_hook

    def save_fwd_hook(self):
        self.vis_conv = None

        def fwd_hook(__, _, output):
            self.vis_conv = output

        return fwd_hook

    def encode_image(self, im, pids=None):

        with torch.no_grad():
            x = self.im_cnn(im).squeeze()

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        if self.use_single_image_head:
            y = self.im_encoder(x)
        else:
            y = torch.zeros(len(im), self.out_dim).to(self.device)
            for t in range(len(self.puzzle_ids)):
                idx = pids == int(self.puzzle_ids[t])
                idx = idx.to(self.device)
                if idx.sum() > 0:
                    y[idx] = get_activation_fn(self.args.run_baseline)(
                        self.im_encoder[int(self.puzzle_ids[t])](x[idx])
                    )

        return y

    def decode_text(self, text):
        get_range = lambda x: range(1, x) if x < 70 else range(x - 70 + 4, x)
        tt = text.cpu()
        text = [
            " ".join(
                [
                    self.vocab.idx2word[int(j)]
                    for j in tt[i][get_range(torch.nonzero(tt[i])[-1])]
                ]
            )
            for i in range(len(tt))
        ]
        return text

    def encode_text(self, text):
        if self.word_embed in ["mbert", "bert"]:
            text = self.decode_text(text)
            q_enc = torch.zeros(len(text), gv.max_qlen, gv.word_dim).to(self.device)
            for ii, tt in enumerate(text):
                q_repr = gv.word_embed(tt)
                q_enc[ii, : min(gv.max_qlen, len(q_repr)), :] = q_repr
            x, (h, _) = self.q_lstm(q_enc.float())
            x = get_activation_fn(self.args.run_baseline)(self.q_MLP(x.mean(1)))

        elif self.word_embed in ["siglip"]:

            text = self.decode_text(text)
            # An encoded seq of tokens for mha in qf layer
            if self.args.qf_layer:
                q_enc = torch.zeros(len(text), gv.max_qlen, gv.word_dim).to(self.device)
                for ii, tt in enumerate(text):
                    q_repr = gv.word_embed(tt)
                    q_enc[ii, : min(gv.max_qlen, len(q_repr)), :] = q_repr

            else:
                # as siglip encodes the sequence
                x = gv.word_embed(text)
                x = get_activation_fn(self.args.run_baseline)(self.q_MLP(x))

        return q_enc.float() if self.args.qf_layer else x

    def seq_decoder(self, decoder, repr):
        """run the LSTM decoder sequentially for k steps"""
        out = [None] * gv.MAX_DECODE_STEPS
        hx = None
        for k in range(int(gv.MAX_DECODE_STEPS)):
            try:
                out[k], hx = decoder(repr, hx)
            except:
                pdb.set_trace()
        return out

    def decode_individual_puzzles(self, repr, pids):
        upids = torch.unique(pids)
        out_reprs = {}
        for t in range(len(upids)):
            idx = pids == upids[t]
            key = str(upids[t].item())
            key_idx = (
                np.where(int(key) == np.array(self.sorted_puzzle_ids))[0][0] + 1
            )  # +1 because we use 1-indexed.
            if upids[t] not in gv.SEQ_PUZZLES:
                out_reprs[int(key)] = self.ans_decoder[key_idx](repr[idx])
            else:
                out_reprs[int(key)] = self.seq_decoder(
                    self.ans_decoder[key_idx], repr[idx]
                )
        return out_reprs

    def forward(self, im, q=None, puzzle_ids=None):
        q_repr = self.encode_text(q)
        im_repr = self.encode_image(im.float(), puzzle_ids).float()

        if not self.args.run_baseline:
            if self.args.qf_layer:
                qf_out = self.qf(im_repr, q_repr)
                qv_repr = self.qv_fusion(self.c([im_repr, q_repr.mean(1), qf_out]))
            else:
                qv_repr = self.qv_fusion(self.c([im_repr, q_repr]))

            qvo_repr = self.decode_individual_puzzles(qv_repr, puzzle_ids)
            return qvo_repr
        else:
            qv_feat = self.qv_fusion(torch.cat([im_repr, q_repr], dim=1))
            qvo_feat = self.decode_individual_puzzles(qv_feat, puzzle_ids)
            return qvo_feat


def load_pretrained_models(args, model_name, model=None):

    if args.test and model is not None:
        model_path = os.path.join(
            args.location,
            "ckpt_%s_%s_%s.pth" % (args.model_name, args.word_embed, args.seed),
        )
        print("test: loading checkpoint %s ..." % (model_path))
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint["net"], strict=True)
        return

    preprocess = None

    if args.model_name in ["resnet50"]:
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT
        model = resnet50(weights=weights)
        preprocess = weights.transforms()
        # Make sure image backbone is frozen
        print(
            f"\n Number trainable params before explicit freezing of image backb  {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        for param in model.parameters():

            param.requires_grad = False

        print(
            f"\n Number trainable params after explicit freezing of image backb  {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    elif args.model_name == "dinov2":
        from transformers import AutoImageProcessor, Dinov2Model

        image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        preprocess = image_processor
        print(
            f"\n Number trainable params before explicit freezing of image backb  {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        for param in model.parameters():

            param.requires_grad = False

        print(
            f"\n Number trainable params after explicit freezing of image backb  {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    elif args.model_name == "siglip":
        from transformers import (
            AutoImageProcessor,
            SiglipVisionModel,
        )

        image_processor = AutoImageProcessor.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
        preprocess = image_processor
        print(
            f"\n Number trainable params before explicit freezing of image backb  {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )
        for param in model.parameters():

            param.requires_grad = False

        print(
            f"\n Number trainable params after explicit freezing of image backb  {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

    elif args.model_name == "fused_dinov2_siglip":
        from transformers import AutoImageProcessor, SiglipVisionModel, Dinov2Model

        model_siglip = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        print(
            f"\n Number trainable params before explicit freezing of image backb  {sum(p.numel() for p in model_siglip.parameters() if p.requires_grad)}"
        )
        for param in model_siglip.parameters():

            param.requires_grad = False

        print(
            f"\n Number trainable params after explicit freezing of image backb  {sum(p.numel() for p in model_siglip.parameters() if p.requires_grad)}"
        )
        model_dino = Dinov2Model.from_pretrained("facebook/dinov2-base")
        model = (model_dino, model_siglip)
        preprocess = None
    else:
        print("model name is %s: not loading pre-trained model." % (args.model_name))

    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint["state_dict"]
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith("module.encoder") and not k.startswith(
                    "module.encoder.fc"
                ):
                    # remove prefix
                    state_dict[k[len("module.encoder.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))
    return model, preprocess
