# https://github.com/Zhudongsheng75/VisLingInstruct/blob/main/mmlm_vicuna.py
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/blip_2/modeling_blip_2.py#L712

import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

# not working with lavis or vicuna.
# go to the from scratch; a small version for proof of concept.
# Q-former inspired layer, but small.
