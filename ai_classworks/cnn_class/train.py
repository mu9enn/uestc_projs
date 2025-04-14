import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm
from model import get_rn50_model, get_clip_linear_probe, get_clip_model_rn50
from torchvision.transforms import ToPILImage
import clip

class CustomDataset(Dataset):
    def __init__(self, X, y, dataset_name="cifar", transform=None):
        self.X = torch.tensor(X).float()
        self.y = torch.tensor(y).long()
        self.dataset_name = dataset_name
        self.transform = transform
        if dataset_name == "cifar":
            self.X = self.X.reshape(-1, 3, 32, 32)  # CIFAR: 3x32x32
        elif dataset_name == "ham10000":
            self.X = self.X.reshape(-1, 1, 28, 28)  # HAM10000: 1x28x28
        self.to_pil = ToPILImage() if transform is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        if self.transform:
            pil_img = self.to_pil(img)
            img = self.transform(pil_img)
        elif self.dataset_name == "ham10000":
            img = img.repeat(3, 1, 1)  # Convert 1-channel to 3-channel for RN50
        return img, self.y[idx]

def train_one_epoch(model, optimizer, criterion, train_loader, device, scaler):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Training loss: {avg_loss:.4f}")
    return avg_loss

def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def clip_zero_shot_classification(model, processor, image, candidate_labels, device):
    """Perform zero-shot classification with CLIP."""
    text_inputs = clip.tokenize(candidate_labels).to(device)
    image_input = processor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        similarities = (image_features @ text_features.T).softmax(dim=-1)
        pred_idx = similarities.argmax().item()
    return candidate_labels[pred_idx]

def evaluate_clip_zeroshot(model, processor, val_loader, candidate_labels, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    to_pil = ToPILImage()
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Zero-shot Evaluating"):
            batch_size = inputs.size(0)
            for i in range(batch_size):
                pil_image = to_pil(inputs[i].cpu())
                pred_label = clip_zero_shot_classification(model, processor, pil_image, candidate_labels, device)
                if candidate_labels.index(pred_label) == labels[i].item():
                    correct += 1
                total += 1
    accuracy = 100 * correct / total
    return accuracy

def perform_10_fold_cv(X, y, dataset_name="cifar", model_type="RN50", pretrained=False, num_epochs=5, batch_size=512, device="cuda", candidate_labels=None):
    """Perform 10-fold cross-validation with specified model and dataset."""
    dataset = CustomDataset(X, y, dataset_name=dataset_name)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []
    num_classes = len(np.unique(y))
    scaler = GradScaler()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f"\nFold {fold + 1}")
        if model_type == "CLIP_linear":
            model, preprocess = get_clip_linear_probe(device=device, num_classes=num_classes)
            dataset_transformed = CustomDataset(X, y, dataset_name=dataset_name, transform=preprocess)
            train_subset = Subset(dataset_transformed, train_idx)
            val_subset = Subset(dataset_transformed, val_idx)
        else:
            if model_type == "RN50":
                model = get_rn50_model(pretrained=pretrained, num_classes=num_classes)
            elif model_type == "CLIP_zeroshot":
                model, processor = get_clip_model_rn50(device=device)
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        if model_type in ["RN50", "CLIP_linear"]:
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}")
                train_one_epoch(model, optimizer, criterion, train_loader, device, scaler)
                scheduler.step()
            accuracy = evaluate_model(model, val_loader, device)
            print(f"Validation Accuracy: {accuracy:.2f}%")
            fold_accuracies.append(accuracy)
        elif model_type == "CLIP_zeroshot":
            if candidate_labels is None:
                raise ValueError("candidate_labels must be provided for CLIP_zeroshot")
            accuracy = evaluate_clip_zeroshot(model, processor, val_loader, candidate_labels, device)
            print(f"Zero-shot Validation Accuracy: {accuracy:.2f}%")
            fold_accuracies.append(accuracy)
        else:
            raise ValueError("Unsupported model_type")

    mean_accuracy = np.mean(fold_accuracies)
    print(f"\nMean Accuracy over 10 folds: {mean_accuracy:.2f}%")
    return fold_accuracies