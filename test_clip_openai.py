
import clip

print(clip.available_models())
model, preprocess = clip.load("RN50x4")
model.eval()
print(model)
# maybe feat size is 640