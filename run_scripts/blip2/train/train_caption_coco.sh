python -m torch.distributed.run --nproc_per_node=16 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml


python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/caption_coco_ft.yaml



python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/caption_lexica_ft.yaml