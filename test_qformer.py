# https://github.com/Zhudongsheng75/VisLingInstruct/blob/main/mmlm_vicuna.py
import logging
import string
from packaging import version

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

import transformers

from mmlm.common.registry import registry
from mmlm.models.blip2 import Blip2Base, disabled_train

# not installing here on mac m1

# note base in ice is python 3.12
# cant 'f
# pip install salesforce-lavis

# will try with conda python version
