#coding: utf-8


import torch
from lavis.models import load_model_and_preprocess

import torch
from PIL import Image


# load sample image



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loads BLIP caption base model, with finetuned checkpoints on MSCOCO captioning dataset.
# this also loads the associated image processors
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
print('here')


# raw_image = Image.open("docs/_static/merlion.png").convert("RGB")

raw_image = Image.open("docs/_static/rooster.jpg").convert("RGB")
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# generate caption
res = model.generate({"image": image})
print("res: {}".format(res))
# ['a large fountain spewing water into the air']