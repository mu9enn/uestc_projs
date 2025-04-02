import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
import numpy as np
from model import get_rn50_model, get_clip_linear_probe, get_clip_model_rn50, clip_zero_shot_classification
from torchvision.transforms import ToPILImage


class CIFAR10Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        # X is a numpy array of shape (n_samples, 3072)
        self.X = torch.tensor(X).float().reshape(-1, 3, 32, 32)
        self.y = torch.tensor(y).long()
        self.transform = transform
        self.to_pil = ToPILImage() if transform is not None else None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.X[idx]
        # If a transform is provided, convert tensor to PIL Image first.
        if self.transform:
            pil_img = self.to_pil(img)
            img = self.transform(pil_img)
        return img, self.y[idx]


def train_one_epoch(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Training loss: {avg_loss:.4f}")
    return avg_loss


def evaluate_model(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


def evaluate_clip_zeroshot(model, processor, val_loader, candidate_labels, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    to_pil = ToPILImage()  # Convert tensors to PIL images
    with torch.no_grad():
        for inputs, labels in val_loader:
            batch_size = inputs.size(0)
            for i in range(batch_size):
                pil_image = to_pil(inputs[i].cpu())
                # Pass device parameter to ensure consistency.
                pred_label, _ = clip_zero_shot_classification(model, processor, pil_image, candidate_labels,
                                                              device=device)
                if candidate_labels.index(pred_label) == labels[i].item():
                    correct += 1
                total += 1
    accuracy = 100 * correct / total
    return accuracy


def perform_10_fold_cv(X, y, model_type="RN50", pretrained=False, num_epochs=5, batch_size=1024, device="cpu"):
    """
    Runs 10-fold cross validation for different experimental conditions.

    model_type options:
      - "RN50": Standard ResNet50 (with or without pre-training).
      - "CLIP_linear": CLIP with linear probing.
      - "CLIP_zeroshot": Zero-shot classification using CLIP.
    """
    # Initially create a dataset without transform.
    dataset = CIFAR10Dataset(X, y)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    # CIFAR-10 candidate labels (order should correspond to numeric labels)
    candidate_labels = ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(np.arange(len(dataset)))):
        print(f"\nFold {fold + 1}")
        # For RN50 and CLIP_zeroshot, use the original dataset;
        # For CLIP_linear, reinitialize with the CLIP preprocess transform.
        if model_type == "CLIP_linear":
            # Get the CLIP linear probe model and the preprocess transform.
            model, preprocess = get_clip_linear_probe(device=device, num_classes=10)
            # Re-create the dataset with the provided transform.
            dataset_transformed = CIFAR10Dataset(X, y, transform=preprocess)
            train_subset = Subset(dataset_transformed, train_idx)
            val_subset = Subset(dataset_transformed, val_idx)
        else:
            if model_type == "RN50":
                model = get_rn50_model(pretrained=pretrained)
            elif model_type == "CLIP_zeroshot":
                model, processor = get_clip_model_rn50(device=device)
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        if model_type == "RN50":
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}")
                train_one_epoch(model, optimizer, criterion, train_loader, device)
            accuracy = evaluate_model(model, val_loader, device)
            print(f"Validation Accuracy: {accuracy:.2f}%")
            fold_accuracies.append(accuracy)

        elif model_type == "CLIP_linear":
            model = model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(num_epochs):
                print(f"Epoch {epoch + 1}/{num_epochs}")
                train_one_epoch(model, optimizer, criterion, train_loader, device)
            accuracy = evaluate_model(model, val_loader, device)
            print(f"Validation Accuracy: {accuracy:.2f}%")
            fold_accuracies.append(accuracy)

        elif model_type == "CLIP_zeroshot":
            # For zero-shot, no training is performed.
            accuracy = evaluate_clip_zeroshot(model, processor, val_loader, candidate_labels, device)
            print(f"Zero-Shot Validation Accuracy: {accuracy:.2f}%")
            fold_accuracies.append(accuracy)
        else:
            raise ValueError("Unsupported model type. Choose from 'RN50', 'CLIP_linear', or 'CLIP_zeroshot'.")

    mean_accuracy = np.mean(fold_accuracies)
    print(f"\nMean Accuracy over 10 folds: {mean_accuracy:.2f}%")
    return fold_accuracies
