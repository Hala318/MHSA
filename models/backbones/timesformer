import torch
from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn

# Load the Timesformer model
model = TimesformerForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k600")

# Access the backbone of the model
backbone = model.timesformer


class TimesformerFeatureExtractor(nn.Module):
    def __init__(self, backbone):
        super(TimesformerFeatureExtractor, self).__init__()
        self.backbone = backbone

    def forward(self, pixel_values, **kwargs):
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        sequence_output = outputs.last_hidden_state[:, 0]  # Assuming you want the representation of the first token
        return sequence_output

# Create the feature extractor model
feature_extractor = TimesformerFeatureExtractor(backbone)

# Save the modified model without the classification head
torch.save(feature_extractor.state_dict(), "timesformer_feature_extractor.pth")

def timesFormer(pretrained: bool = False, in_channels: int = 1, **kwargs: any) -> TimesformerFeatureExtractor:
    model = feature_extractor(
        **kwargs,
    )
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls["MobileNet"])
    #     state_dict = _modify_weights(state_dict, in_channels)
    #     model.load_state_dict(state_dict, strict=True)

    return model
