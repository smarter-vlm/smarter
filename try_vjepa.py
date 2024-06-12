# vjepa large
# no official hf; need to install jepa folder 
# siglip conda env
# can't install one package on M1 - decoder 

# https://huggingface.co/nielsr/vit-large-patch16-v-jepa

from src.models.vision_transformer import VisionTransformer

model = VisionTransformer.from_pretrained("nielsr/vit-large-patch16-v-jepa")
print(model)
# try from FB directly https://github.com/facebookresearch/jepa 