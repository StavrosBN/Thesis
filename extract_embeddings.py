import os
import numpy as np
import torch

from models.model import ConvTran

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CHECKPOINTS = {
    'cleaned_1': 'outputs/models/convtran_cleaned_1.pt',
    'cleaned_2': 'outputs/models/convtran_cleaned_2.pt',
    'cleaned_3': 'outputs/models/convtran_cleaned_3.pt',
}

FULL_DATA_DIR = 'data/windows/full'
OUTPUT_DIR = 'outputs/embeddings'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(checkpoint_path):
    """Ξαναφτιάχνει ένα ConvTran ακριβώς όπως ήταν στην εκπαίδευση,
    χρησιμοποιώντας το config που αποθηκεύτηκε μέσα στο checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = ConvTran(
        num_features=ckpt['num_features'],
        seq_len=ckpt['seq_len'],
        num_classes=ckpt['num_classes'],
        emb_size=ckpt['emb_size'],
        num_heads=ckpt['num_heads'],
        dim_ff=ckpt['dim_ff'],
        dropout=ckpt['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()  # ΚΡΙΣΙΜΟ: απενεργοποιεί dropout/batchnorm updates, καθαρό inference
    return model


def extract_combined_embeddings(models, X):
    """
    X: (N, 60, 30) -- ίδιο σχήμα με τα windows.
    Επιστρέφει (N, 3*emb_size) -- τα embeddings των 3 μοντέλων ενωμένα.
    """
    X = X.astype(np.float32).transpose(0, 2, 1)  # (N, 30, 60), όπως θέλει το ConvTran
    X_tensor = torch.from_numpy(X).to(DEVICE)

    embeddings_per_model = []
    with torch.no_grad():  # καμία εκπαίδευση, μόνο forward pass
        for model in models:
            emb = model(X_tensor, return_embedding=True)  # (N, emb_size)
            embeddings_per_model.append(emb.cpu().numpy())

    combined = np.concatenate(embeddings_per_model, axis=1)  # (N, 3*emb_size)
    return combined


if __name__ == '__main__':
    print(f'Χρήση συσκευής: {DEVICE}')

    # ---- Φόρτωση των 3 ήδη εκπαιδευμένων μοντέλων ----
    models = []
    for name, path in CHECKPOINTS.items():
        print(f'Φόρτωση {name} από {path}')
        models.append(load_model(path))

    # ---- Εξαγωγή embeddings για meta_train και test ----
    for split_name in ['meta_train', 'test']:
        data = np.load(os.path.join(FULL_DATA_DIR, f'{split_name}.npz'))
        X, y = data['X'], data['y']

        combined = extract_combined_embeddings(models, X)

        out_path = os.path.join(OUTPUT_DIR, f'{split_name}.npz')
        np.savez(out_path, X=combined, y=y)
        print(f'{out_path}: X={combined.shape}, y={y.shape}')