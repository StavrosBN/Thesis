import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from models.model import ConvTran

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_PATH = 'outputs/models/convtran_single_full.pt'
TEST_DATA_PATH = 'data/windows/full/test.npz'
OUTPUT_PLOTS_DIR = 'outputs/plots'
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def load_model(checkpoint_path):
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
    model.eval()
    return model


if __name__ == '__main__':
    model = load_model(CHECKPOINT_PATH)

    data = np.load(TEST_DATA_PATH)
    X, y_test = data['X'], data['y']
    X = X.astype(np.float32).transpose(0, 2, 1)  # (N, 60, 30) -> (N, 30, 60)
    X_tensor = torch.from_numpy(X).to(DEVICE)

    with torch.no_grad():
        logits = model(X_tensor)  # ΚΑΝΟΝΙΚΗ πρόβλεψη τώρα, όχι embedding
        y_pred = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, y_pred)
    print('=== Single ConvTran (χωρίς ensemble), πλήρες πρόβλημα 10 κλάσεων ===')
    print(f'Test accuracy: {acc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Πρόβλεψη')
    plt.ylabel('Πραγματική κλάση')
    plt.title(f'Single ConvTran Confusion Matrix (test accuracy={acc:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'single_convtran_confusion_matrix.png'), dpi=150)
    print(f'\nΑποθηκεύτηκε: {OUTPUT_PLOTS_DIR}/single_convtran_confusion_matrix.png')