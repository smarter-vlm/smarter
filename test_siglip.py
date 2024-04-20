from PIL import Image
import requests
from transformers import (
    AutoProcessor,
    SiglipVisionModel,
    SiglipTextModel,
    AutoTokenizer,
)

model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state
pooled_output = outputs.pooler_output  # pooled features #TODO DR: do my own pooling

print("image model", model)
print("frozen vision output", pooled_output)

model = SiglipTextModel.from_pretrained("google/siglip-base-patch16-224")
tokenizer = AutoTokenizer.from_pretrained("google/siglip-base-patch16-224")

inputs = tokenizer(
    ["a photo of a cat", "a photo of a dog"], padding="max_length", return_tensors="pt"
)

outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

pooled_output = outputs.pooler_output

# text model
print("text model", model)
print("text pooled shape", pooled_output.shape)
