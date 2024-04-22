import os
import warnings

import torch
import torch.nn as nn

warnings.filterwarnings("ignore")
import pdb
import pickle

import clip
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import models

import text_encoder as gv


class Smarter_VL(nn.Module):
    def __init__(self, args, VL_backbone):
        super(Smarter_VL, self).__init__()
        vocab_path = args.vocab_path
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.num_opts = 5
        self.out_dim = args.repr_size
        self.h_sz = 256
        self.repr_size = 768
        self.dummy_question = None
        self.model_name = args.model_name
        self.use_clip_text = args.use_clip_text
        self.use_single_image_head = args.use_single_image_head
        self.sorted_puzzle_ids = np.sort(np.array([int(ii) for ii in args.puzzle_ids]))

        self.max_val = gv.MAX_VAL + 1

        self.processor = args.preprocess
        self.VL_backbone = VL_backbone
        self.create_puzzle_head(args)

        self.q_MLP = nn.Sequential(
            nn.Linear(self.repr_size, self.h_sz),
            nn.GELU(),
            nn.Linear(self.h_sz, self.out_dim),
            nn.GELU(),
        )

        self.qv_MLP = nn.Sequential(
            nn.Linear(self.repr_size, self.h_sz),
            nn.GELU(),
            nn.Linear(self.h_sz, self.out_dim),
            nn.GELU(),
        )

        self.qv_fusion = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.GELU(),
        )

        self.create_puzzle_tail(args)

    def create_puzzle_head(self, args):
        if args.use_single_image_head:
            self.im_encoder = nn.Sequential(
                nn.Linear(self.repr_size, self.out_dim),
                nn.GELU(),
                nn.Linear(self.out_dim, self.out_dim),
            )
        else:
            self.puzzle_ids = args.puzzle_ids
            im_encoder = [nn.Sequential(nn.Linear(self.out_dim, 1))]
            for i in range(1, gv.num_puzzles + 1):
                im_encoder.append(
                    nn.Sequential(
                        nn.Linear(self.repr_size, self.out_dim),
                        nn.GELU(),
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
                ans_decoder.append(
                    nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.GELU(),
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.GELU(),
                        nn.Linear(self.out_dim, num_classes),
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

    def process(self, images, text):
        inputs = self.processor(
            text=text,
            images=images,
            return_tensors="pt",
            max_length=77,
            padding=True,
            return_codebook_pixels=True,
            return_image_mask=True,
        )
        inputs["input_ids_masked"] = inputs["input_ids"].detach().clone()
        inputs["bool_masked_pos"] = torch.zeros_like(inputs["bool_masked_pos"])
        inputs = inputs.to("cuda")
        return inputs

    def encode_image(self, im_repr, pids=None):
        if self.use_single_image_head:
            y = self.im_encoder(im_repr)
        else:
            y = torch.zeros(len(im_repr), im_repr.shape[1], self.out_dim).cuda()
            for t in range(len(self.puzzle_ids)):
                idx = pids == int(self.puzzle_ids[t])
                idx = idx.cuda()
                if idx.sum() > 0:
                    y[idx] = F.relu(
                        self.im_encoder[int(self.puzzle_ids[t])](im_repr[idx])
                    )
        return y

    def encode_image_and_text(self, qv_repr):
        x = F.relu(self.qv_MLP(qv_repr))
        return x

    def encode_text(self, q_repr):
        x = F.relu(self.q_MLP(q_repr))
        return x

    def decode_image(self, im_list):
        im_list = (im_list.permute(0, 2, 3, 1) * 255).cpu().numpy().astype("uint8")
        im_list = [
            Image.fromarray(im_list[ii]) for ii in range(len(im_list))
        ]  # convert im
        return im_list

    def decode_text(self, text):
        tt = text.cpu()
        text = [
            " ".join(
                [
                    self.vocab.idx2word[int(j)]
                    for j in tt[i][1 : torch.nonzero(tt[i])[-1]]
                ]
            )
            for i in range(len(tt))
        ]
        return text

    def seq_decoder(self, decoder, repr):
        """run the LSTM decoder sequentially for k steps"""
        out = [None] * gv.MAX_DECODE_STEPS
        hx = None
        for k in range(gv.MAX_DECODE_STEPS):
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
        im = self.decode_image(im)
        q_text = self.decode_text(q)
        inputs = self.process(im, q_text)

        with torch.no_grad():
            outputs = self.VL_backbone(**inputs)

        im_repr = (
            outputs.image_embeddings
        )  # Batch size X (Number of image patches + 1) x Hidden size => 2 X 197 X 768
        q_repr = (
            outputs.text_embeddings
        )  # Batch size X (Text sequence length + 1) X Hidden size => 2 X 77 X 768

        im_repr = self.encode_image(im_repr.float(), puzzle_ids)
        q_repr = self.encode_text(q_repr)

        qv_repr = self.qv_fusion(torch.cat([im_repr.mean(1), q_repr.mean(1)], dim=1))

        qvo_repr = self.decode_individual_puzzles(qv_repr, puzzle_ids)

        return qvo_repr


class Puzzle_Net(nn.Module):
    def __init__(self, args, im_backbone=None):
        super(Puzzle_Net, self).__init__()
        vocab_path = args.vocab_path
        with open(vocab_path, "rb") as f:
            self.vocab = pickle.load(f)

        self.num_opts = 5
        self.out_dim = args.repr_size
        self.h_sz = 256
        self.dummy_question = None
        self.model_name = args.model_name
        self.use_clip_text = args.use_clip_text
        self.use_single_image_head = args.use_single_image_head
        self.word_embed = args.word_embed
        self.sorted_puzzle_ids = np.sort(np.array([int(ii) for ii in args.puzzle_ids]))

        self.max_val = gv.MAX_VAL + 1

        # image backbones.
        if args.model_name[:6] == "resnet":
            self.im_repr_size = im_backbone.fc.weight.shape[1]
            modules = list(im_backbone.children())[:-1]
            self.im_cnn = nn.Sequential(*modules)

        elif args.model_name in ["mae"]:
            self.preprocess = args.preprocess
            self.im_cnn = lambda x: self.process_MAE(x)
            self.im_backbone = im_backbone
            self.im_repr_size = 768

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
            self.im_cnn = lambda x: self.process_fused(
                x, image_processor_siglip, image_processor_dino
            )
            self.im_backbone = im_backbone
            self.im_repr_size = 768 + 768

        else:
            raise "unknown model_name %s" % (args.model_name)

        self.create_puzzle_head(args)

        # language backbones
        if self.use_clip_text:
            self.q_encoder, _ = clip.load("ViT-B/32", device="cuda")
            self.clip_dim = 512
            self.q_MLP = nn.Sequential(
                nn.Linear(self.clip_dim, self.h_sz),
                nn.GELU(),
                nn.Linear(self.h_sz, self.out_dim),
            )
        else:
            if args.word_embed == "standard":
                self.q_emb = nn.Embedding(len(self.vocab), self.h_sz, max_norm=1)
                self.q_lstm = nn.LSTM(
                    int(self.h_sz),
                    int(self.h_sz),
                    num_layers=2,
                    batch_first=True,
                    bidirectional=True,
                )
            else:
                word_dim = gv.word_dim
                self.q_emb = nn.Identity()
                self.q_lstm = nn.GRU(
                    int(word_dim),
                    int(self.h_sz),
                    num_layers=1,
                    batch_first=True,
                    bidirectional=True,
                    bias=False,
                )
            self.q_MLP = nn.Linear(self.h_sz * 2, self.out_dim)

        self.o_encoder = nn.Sequential(
            nn.Embedding(len(self.vocab), self.out_dim, max_norm=1),
            nn.Linear(self.out_dim, self.out_dim),
            nn.GELU(),
        )
        self.qv_fusion = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.GELU(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.GELU(),
        )

        self.create_puzzle_tail(args)

    def process_MAE(self, x):
        x = self.decode_image(x)  # get from tensor to PIL images
        inputs = self.preprocess(images=x, return_tensors="pt").to("cuda")
        outputs = self.im_backbone(**inputs)
        return outputs.last_hidden_state.mean(1)

    def process_dinov2(self, x):
        device = torch.device("cuda")
        x = self.decode_image(x)
        inputs = self.preprocess(images=x, do_rescale=True, return_tensors="pt").to(
            device
        )
        outputs = self.im_backbone(**inputs)
        return outputs.last_hidden_state.mean(1)

    def process_fused(self, x, image_processor_siglip, image_processor_dino):
        device = torch.device("cuda")
        x = self.decode_image(x)

        inputs_din = image_processor_dino(
            images=x, do_rescale=True, return_tensors="pt"
        ).to(device)
        inputs_sig = image_processor_siglip(
            images=x, do_rescale=True, return_tensors="pt"
        ).to(device)

        im_backbone_din, im_backbone_sig = self.im_backbone

        im_backbone_din = im_backbone_din.to(device)
        im_backbone_sig = im_backbone_sig.to(device)

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
                nn.GELU(),
                nn.Linear(self.out_dim, self.out_dim),
            )
        else:
            self.puzzle_ids = args.puzzle_ids
            im_encoder = [nn.Sequential(nn.Linear(self.out_dim, 1))]
            for i in range(1, gv.num_puzzles + 1):
                im_encoder.append(
                    nn.Sequential(
                        nn.Linear(self.im_repr_size, self.out_dim),
                        nn.GELU(),
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
        for pid in puzzles:  # self.puzzle_ids:
            num_classes = gv.NUM_CLASSES_PER_PUZZLE[str(pid)]
            if int(pid) not in gv.SEQ_PUZZLES:
                ans_decoder.append(
                    nn.Sequential(
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.GELU(),
                        nn.Linear(self.out_dim, self.out_dim),
                        nn.GELU(),
                        nn.Linear(self.out_dim, num_classes),
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
            y = torch.zeros(len(im), self.out_dim).cuda()
            for t in range(len(self.puzzle_ids)):
                idx = pids == int(self.puzzle_ids[t])
                idx = idx.cuda()
                if idx.sum() > 0:
                    y[idx] = F.relu(self.im_encoder[int(self.puzzle_ids[t])](x[idx]))

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
        if self.word_embed == "standard":
            x = self.q_emb(text)
            x, (h, _) = self.q_lstm(x.float())
            x = F.relu(self.q_MLP(x.mean(1)))
        elif self.word_embed == "bert":
            text = self.decode_text(text)
            q_enc = torch.zeros(len(text), gv.max_qlen, gv.word_dim).cuda()
            for ii, tt in enumerate(text):
                q_repr = gv.word_embed(tt)
                q_enc[ii, : min(gv.max_qlen, len(q_repr)), :] = q_repr
            x, (h, _) = self.q_lstm(q_enc.float())
            x = F.relu(self.q_MLP(x.mean(1)))
        else:
            x = gv.word_embed(text)

        return x

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

        qv_repr = self.qv_fusion(torch.cat([im_repr, q_repr], dim=1))

        qvo_repr = self.decode_individual_puzzles(qv_repr, puzzle_ids)
        return qvo_repr


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

    elif args.model_name == "clip":
        model, preprocess = clip.load("ViT-B/32", device="cuda")

    elif args.model_name == "mae":
        from transformers import AutoFeatureExtractor, ViTMAEModel

        repr_extractor = AutoFeatureExtractor.from_pretrained("facebook/vit-mae-base")
        model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
        preprocess = repr_extractor

    elif args.model_name == "dinov2":
        from transformers import AutoImageProcessor, Dinov2Model

        image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = Dinov2Model.from_pretrained("facebook/dinov2-base")
        preprocess = image_processor

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

    elif args.model_name == "fused_dinov2_siglip":

        from transformers import AutoImageProcessor, SiglipVisionModel, Dinov2Model

        # image_processor_siglip = AutoImageProcessor.from_pretrained(
        #     "google/siglip-base-patch16-224"
        # )
        model_siglip = SiglipVisionModel.from_pretrained(
            "google/siglip-base-patch16-224"
        )
        # image_processor_dino = AutoImageProcessor.from_pretrained(
        #     "facebook/dinov2-base"
        # )
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
