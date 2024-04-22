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
# Q-former inspired layer, but small: a qf layer.

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2Model

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2Model.from_pretrained("Salesforce/blip2-opt-2.7b")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")
qformer_outputs = model.get_qformer_features(**inputs)

print(qformer_outputs.last_hidden_state.shape)

print(model)
