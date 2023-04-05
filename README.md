<p align="center">
    <br>
    <img src="docs/_static/logo_final.png" width="800"/>
    <br>
<p>

# Image2Pormpt- Make it easy to write pormpts

## Base Model：BLIP2 in LAVIS

* [Model Release] Jan 2023, released implementation of **BLIP-2** <br>
  [Paper](https://arxiv.org/abs/2301.12597), [Project Page](https://github.com/salesforce/LAVIS/tree/main/projects/blip2), [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/LAVIS/blob/main/examples/blip2_instructed_generation.ipynb)

> A generic and efficient pre-training strategy that easily harvests development of pretrained vision models and large language models (LLMs) for vision-language pretraining. BLIP-2 beats Flamingo on zero-shot VQAv2 (**65.0** vs **56.3**), establishing new state-of-the-art on zero-shot captioning (on NoCaps **121.6** CIDEr score vs previous best **113.2**). In addition, equipped with powerful LLMs (e.g. OPT, FlanT5), BLIP-2 also unlocks the new **zero-shot instructed vision-to-language generation** capabilities for various interesting applications!

## Introduction

Image-to-prompt is a Python deep learning model for generating prompt from image for text-to-image tasks.

## Installation

1. (Optional) Creating conda environment

```bash
conda create -n lavis python=3.8
conda activate lavis
```

2. install from [PyPI](https://pypi.org/project/salesforce-lavis/)

```bash
pip install salesforce-lavis
```

3. Or, for development, you may build from source

```bash
git clone https://github.com/salesforce/LAVIS.git
cd LAVIS
pip install -e .
```

## Getting Started

### Model Zoo

Model are in google drive, to view:

```model
https://drive.google.com/drive/u/0/folders/14C3xykU_HbXUFv_akJ2uVD-sgX75gi4m
```

### Image Captioning

In this example, we use the BLIP model to generate a prompt for the image. To make inference even easier, we also associate each
pre-trained model with its preprocessors (transforms), accessed via ``load_model_and_preprocess()``.

model_path are modified in

```model
lavis/configs/models/blip2/blip2_caption_opt2.7b.yaml
finetuned: local_path
```

```python inference.py
import torch
from lavis.models import load_model_and_preprocess
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, vis_processors, _ = load_model_and_preprocess(name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device)
raw_image = Image.open("docs/_static/rooster.jpg").convert("RGB")
# preprocess the image
# vis_processors stores image transforms for "train" and "eval" (validation / testing / inference)
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
# generate caption
res = model.generate({"image": image})
print("res: {}".format(res))
#['rooster in oriental armor pattern, kung fu style, intricate, high resolution, art style, kirby, kirby art,']
```

```## Contact us

If you have any questions, comments or suggestions, please do not hesitate to contact us at 976650543@qq.com.

## License

[BSD 3-Clause License](LICENSE.txt)
