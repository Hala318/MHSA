import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from transformers import VivitImageProcessor, VivitForVideoClassification

model = VivitForVideoClassification.from_pretrained("google/vivit-b-16x2-kinetics400")

backbone = model.vivit

def timesFormer(pretrained: bool = False,  **kwargs: any):
    # b = backbone(
    #     **kwargs,
    # )
    config = model.config

    # Check for the number of frames
    num_frames = config.num_frames
    print(f"The model expects {num_frames} frames per video clip.")
    return backbone
