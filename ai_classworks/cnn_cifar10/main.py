import torch
from data_preprocessing import load_data, preprocess_data
from train import perform_10_fold_cv


def main():
    df = load_data()
    X, y = preprocess_data(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)

    print("Experiment 1: RN50 without pre-training")
    perform_10_fold_cv(X, y, model_type="RN50", pretrained=False, num_epochs=20, device=device)

    print("\nExperiment 2: RN50 with pre-training")
    perform_10_fold_cv(X, y, model_type="RN50", pretrained=True, num_epochs=20, device=device)

    print("\nExperiment 3: CLIP zero-shot reasoning")
    perform_10_fold_cv(X, y, model_type="CLIP_zeroshot", device=device)

    print("\nExperiment 4: CLIP linear probing")
    perform_10_fold_cv(X, y, model_type="CLIP_linear", num_epochs=20, device=device)


if __name__ == "__main__":
    main()
