import torch
import torch.nn as nn
import torchvision.models as models
import clip
from PIL import Image


def get_rn50_model(pretrained=False):
    """
    Returns a standard ResNet50 model adjusted for CIFAR-10.
    """
    model = models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def get_clip_model_rn50(device="cpu"):
    """
    Loads the CLIP model (RN50 variant) and its preprocess function using the clip package.
    """
    model, preprocess = clip.load("RN50", device=device)
    return model, preprocess


def clip_zero_shot_classification(model, preprocess, image, candidate_labels, device="cpu"):
    """
    Performs zero-shot classification with CLIP:
      - image: a PIL Image.
      - candidate_labels: list of label strings.

    Returns:
      A tuple (predicted_label, probs) where probs is a tensor of softmax probabilities.
    """
    # Tokenize candidate labels
    text = clip.tokenize(candidate_labels).to(device)
    # Preprocess the image (preprocess includes resizing, normalization, etc.)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        # Forward the model to obtain logits
        logits_per_image, _ = model(image_input, text)
        probs = logits_per_image.softmax(dim=-1)
        pred_idx = probs.argmax().item()
    return candidate_labels[pred_idx], probs


def get_clip_linear_probe(device="cpu", num_classes=10):
    """
    Sets up a CLIP linear probing model.
    - Loads the CLIP RN50 model.
    - Freezes its parameters.
    - Adds a trainable linear classifier on top of the image encoder output.

    Returns:
      (linear_probe_model, preprocess)
    """
    # Load CLIP RN50 variant
    model, preprocess = clip.load("RN50", device=device)
    model.eval()
    # Freeze CLIP parameters
    for param in model.parameters():
        param.requires_grad = False

    # For RN50, the image features have 1024 dimensions (for CLIP).
    output_dim = 1024
    classifier = nn.Linear(output_dim, num_classes)

    class CLIPLinearProbe(nn.Module):
        def __init__(self, clip_model, classifier):
            super().__init__()
            self.clip_model = clip_model
            self.classifier = classifier

        def forward(self, x):
            # x should be preprocessed to match CLIP's expected input (e.g. 224x224)
            with torch.no_grad():
                # Use the CLIP image encoder to get features
                features = self.clip_model.encode_image(x)
            logits = self.classifier(features)
            return logits

    linear_probe_model = CLIPLinearProbe(model, classifier)
    return linear_probe_model, preprocess
