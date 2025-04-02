import torch
import torch.nn as nn
import torchvision.models as models
import clip
from PIL import Image

def get_rn50_model(pretrained=False, num_classes=10):
    """Returns a ResNet50 model adjusted for the specified number of classes."""
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def get_clip_model_rn50(device="cpu"):
    """Loads the CLIP RN50 model and its preprocess function."""
    model, preprocess = clip.load("RN50", device=device)
    return model, preprocess

def get_clip_linear_probe(device="cpu", num_classes=10):
    """Sets up a CLIP linear probing model with a custom number of classes."""
    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    output_dim = 1024  # CLIP RN50 image feature dimension
    classifier = nn.Linear(output_dim, num_classes)

    class CLIPLinearProbe(nn.Module):
        def __init__(self, clip_model, classifier):
            super().__init__()
            self.clip_model = clip_model
            self.classifier = classifier

        def forward(self, x):
            with torch.no_grad():
                features = self.clip_model.encode_image(x)
            features = features.to(torch.float32)
            logits = self.classifier(features)
            return logits

    linear_probe_model = CLIPLinearProbe(model, classifier)
    return linear_probe_model, preprocess