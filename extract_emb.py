import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.model import ConvTran


# -------------------- Ρυθμίσεις --------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

GROUPS = {
    "cleaned_1": {
        "num_classes": 3,
        "data_dir": "data/windows/cleaned_1",
    },
    "cleaned_2": {
        "num_classes": 3,
        "data_dir": "data/windows/cleaned_2",
    },
    "cleaned_3": {
        "num_classes": 4,
        "data_dir": "data/windows/cleaned_3",
    },
}

MODEL_DIR = "outputs/models"
OUTPUT_DIR = "outputs/embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------- Dataset --------------------

class WindowDataset(Dataset):

    def __init__(self, npz_path):

        data = np.load(npz_path)

        X = data["X"].astype(np.float32)

        # ConvTran input:
        # (samples, seq_len, features)
        # ->
        # (samples, features, seq_len)

        X = X.transpose(0, 2, 1)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(data["y"].astype(np.int64))


    def __len__(self):
        return len(self.y)


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



# -------------------- Embedding Extraction --------------------

def extract_embeddings(model, loader):

    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)

            # παίρνουμε το embedding πριν τον classifier
            emb = model(
                X,
                return_embedding=True
            )

            embeddings.append(
                emb.cpu().numpy()
            )

            labels.append(
                y.numpy()
            )


    embeddings = np.concatenate(
        embeddings,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    return embeddings, labels



# -------------------- Main --------------------

if __name__ == "__main__":

    print(f"Using device: {DEVICE}")


    # Τα πραγματικά ονόματα των αρχείων σου
    SPLIT_FILES = {
        "train": "convtran_train.npz",
        "val": "convtran_val.npz",
        "test": "test.npz",
    }


    for group_name, cfg in GROUPS.items():

        print(f"\n===== {group_name} =====")


        # -------- Load trained ConvTran --------

        model_path = os.path.join(
            MODEL_DIR,
            f"convtran_{group_name}.pt"
        )


        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Δεν βρέθηκε το μοντέλο: {model_path}"
            )


        checkpoint = torch.load(
            model_path,
            map_location=DEVICE
        )


        model = ConvTran(
            num_features=checkpoint["num_features"],
            seq_len=checkpoint["seq_len"],
            num_classes=checkpoint["num_classes"],
            emb_size=checkpoint["emb_size"],
            num_heads=checkpoint["num_heads"],
            dim_ff=checkpoint["dim_ff"],
            dropout=checkpoint["dropout"],
        ).to(DEVICE)


        model.load_state_dict(
            checkpoint["model_state_dict"]
        )


        model.eval()


        # -------- Output folder --------

        group_output = os.path.join(
            OUTPUT_DIR,
            group_name
        )

        os.makedirs(
            group_output,
            exist_ok=True
        )


        # -------- Extract all splits --------

        for split, filename in SPLIT_FILES.items():


            npz_path = os.path.join(
                cfg["data_dir"],
                filename
            )


            if not os.path.exists(npz_path):

                raise FileNotFoundError(
                    f"Δεν βρέθηκε dataset αρχείο: {npz_path}"
                )


            dataset = WindowDataset(
                npz_path
            )


            loader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
            )


            X_emb, y = extract_embeddings(
                model,
                loader
            )


            save_path = os.path.join(
                group_output,
                f"{split}_embeddings.npz"
            )


            np.savez(
                save_path,
                X=X_emb,
                y=y
            )


            print(
                f"{split}: embeddings shape = {X_emb.shape}"
            )


    print("\nEmbedding extraction completed successfully.")