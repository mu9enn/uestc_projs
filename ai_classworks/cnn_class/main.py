import torch
from data_preprocessing import load_data, preprocess_data
from train import perform_10_fold_cv

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # CIFAR-10
    print("\n=== CIFAR-10 Experiments ===")
    df_cifar = load_data(dataset_name="cifar")
    X_cifar, y_cifar = preprocess_data(df_cifar, dataset_name="cifar")
    cifar_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    print("\nExperiment 1: RN50 without pre-training (CIFAR-10)")
    perform_10_fold_cv(X_cifar, y_cifar, dataset_name="cifar", model_type="RN50", pretrained=False, num_epochs=20, device=device)

    print("\nExperiment 2: RN50 with pre-training (CIFAR-10)")
    perform_10_fold_cv(X_cifar, y_cifar, dataset_name="cifar", model_type="RN50", pretrained=True, num_epochs=20, device=device)

    print("\nExperiment 3: CLIP linear probing (CIFAR-10)")
    perform_10_fold_cv(X_cifar, y_cifar, dataset_name="cifar", model_type="CLIP_linear", num_epochs=20, device=device)

    print("\nExperiment 4: CLIP zero-shot (CIFAR-10)")
    perform_10_fold_cv(X_cifar, y_cifar, dataset_name="cifar", model_type="CLIP_zeroshot", candidate_labels=cifar_labels, device=device)

    # HAM10000
    print("\n=== HAM10000 Experiments ===")
    df_ham = load_data(dataset_name="ham10000", file_path="hmnist_28_28_L.csv")
    X_ham, y_ham = preprocess_data(df_ham, dataset_name="ham10000")
    # Placeholder labels; replace with actual HAM10000 class names
    ham_labels = [f"class_{i}" for i in range(len(np.unique(y_ham)))]

    print("\nExperiment 1: RN50 without pre-training (HAM10000)")
    perform_10_fold_cv(X_ham, y_ham, dataset_name="ham10000", model_type="RN50", pretrained=False, num_epochs=20, device=device)

    print("\nExperiment 2: RN50 with pre-training (HAM10000)")
    perform_10_fold_cv(X_ham, y_ham, dataset_name="ham10000", model_type="RN50", pretrained=True, num_epochs=20, device=device)

    print("\nExperiment 3: CLIP linear probing (HAM10000)")
    perform_10_fold_cv(X_ham, y_ham, dataset_name="ham10000", model_type="CLIP_linear", num_epochs=20, device=device)

    print("\nExperiment 4: CLIP zero-shot (HAM10000)")
    perform_10_fold_cv(X_ham, y_ham, dataset_name="ham10000", model_type="CLIP_zeroshot", candidate_labels=ham_labels, device=device)

if __name__ == "__main__":
    main()