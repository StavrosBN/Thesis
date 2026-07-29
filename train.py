import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models.model import ConvTran

# ---------- Ρυθμίσεις ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
PATIENCE = 15          # early stopping: σταματάει αν δεν βελτιωθεί το val loss για τόσα epochs
EMB_SIZE = 16
NUM_HEADS = 8
DIM_FF = 256
DROPOUT = 0.2

GROUPS = {
    'cleaned_1': {'num_classes': 3, 'data_dir': 'data/windows/cleaned_1'},
    'cleaned_2': {'num_classes': 3, 'data_dir': 'data/windows/cleaned_2'},
    'cleaned_3': {'num_classes': 4, 'data_dir': 'data/windows/cleaned_3'},
}

OUTPUT_DIR = 'outputs/models'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- Dataset wrapper ----------
class WindowDataset(Dataset):
    """
    Φορτώνει ένα .npz αρχείο (X, y) και κάνει το transpose που χρειάζεται
    το ConvTran: από (N, seq_len, features) -> (N, features, seq_len).
    """
    def __init__(self, npz_path):
        data = np.load(npz_path)
        X = data['X'].astype(np.float32)
        X = X.transpose(0, 2, 1)  # (N, 60, 30) -> (N, 30, 60)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(data['y'].astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- Εκπαίδευση ενός ConvTran ----------
def train_one_group(group_name, num_classes, data_dir):
    print(f'\n=== Εκπαίδευση ConvTran για {group_name} ===')

    train_ds = WindowDataset(os.path.join(data_dir, 'convtran_train.npz'))
    val_ds = WindowDataset(os.path.join(data_dir, 'convtran_val.npz'))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    num_features = train_ds.X.shape[1]  # 30
    seq_len = train_ds.X.shape[2]       # 60

    model = ConvTran(
        num_features=num_features,
        seq_len=seq_len,
        num_classes=num_classes,
        emb_size=EMB_SIZE,
        num_heads=NUM_HEADS,
        dim_ff=DIM_FF,
        dropout=DROPOUT
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
        train_loss /= len(train_ds)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                logits = model(X_batch)
                loss = loss_fn(logits, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                correct += (logits.argmax(dim=1) == y_batch).sum().item()

        val_loss /= len(val_ds)
        val_acc = correct / len(val_ds)

        print(f'Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}')

        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state = {
                'model_state_dict': model.state_dict(),
                'num_features': num_features,
                'seq_len': seq_len,
                'num_classes': num_classes,
                'emb_size': EMB_SIZE,
                'num_heads': NUM_HEADS,
                'dim_ff': DIM_FF,
                'dropout': DROPOUT,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'epoch': epoch,
            }
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f'Early stopping στο epoch {epoch} (καλύτερο ήταν το {best_state["epoch"]})')
                break

    save_path = os.path.join(OUTPUT_DIR, f'convtran_{group_name}.pt')
    torch.save(best_state, save_path)
    print(f'Αποθηκεύτηκε: {save_path} (val_loss={best_state["val_loss"]:.4f}, val_acc={best_state["val_acc"]:.4f})')


# ---------- Εκπαίδευση και των 3 ομάδων ----------
if __name__ == '__main__':
    print(f'Χρήση συσκευής: {DEVICE}')
    for group_name, cfg in GROUPS.items():
        train_one_group(group_name, cfg['num_classes'], cfg['data_dir'])