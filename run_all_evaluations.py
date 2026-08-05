import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, classification_report

from models.model import ConvTran

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

GROUP_CLASS_NAMES = {
    'cleaned_1': ['A', 'B', 'C'],
    'cleaned_2': ['D', 'E', 'F'],
    'cleaned_3': ['G', 'H', 'I', 'J'],
}
ALL_CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def load_convtran(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = ConvTran(
        num_features=ckpt['num_features'], seq_len=ckpt['seq_len'],
        num_classes=ckpt['num_classes'], emb_size=ckpt['emb_size'],
        num_heads=ckpt['num_heads'], dim_ff=ckpt['dim_ff'], dropout=ckpt['dropout'],
    ).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model


def predict_convtran(model, X):
    X = X.astype(np.float32).transpose(0, 2, 1)  # (N, 60, 30) -> (N, 30, 60)
    X_tensor = torch.from_numpy(X).to(DEVICE)
    with torch.no_grad():
        return model(X_tensor).argmax(dim=1).cpu().numpy()


def summarize_windows(X):
    return np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1)


def print_section(title):
    print('\n' + '=' * 70)
    print(title)
    print('=' * 70)


if __name__ == '__main__':
    results = {}  # για τον τελικό συγκεντρωτικό πίνακα

    # ---- 1. Τα 3 επιμέρους ConvTran, στο δικό τους πραγματικό test set ----
    print_section('1. ΕΠΙΜΕΡΟΥΣ ConvTran (test set ανά ομάδα)')
    for group in ['cleaned_1', 'cleaned_2', 'cleaned_3']:
        model = load_convtran(f'outputs/models/convtran_{group}.pt')
        data = np.load(f'data/windows/{group}/test.npz')
        y_pred = predict_convtran(model, data['X'])
        acc = accuracy_score(data['y'], y_pred)
        results[f'ConvTran ({group})'] = acc
        print(f'\n{group}: test accuracy = {acc:.4f}')
        print(classification_report(data['y'], y_pred, target_names=GROUP_CLASS_NAMES[group], zero_division=0))

    # ---- 2. Single ConvTran, απευθείας στο πλήρες πρόβλημα (χωρίς ensemble) ----
    print_section('2. SINGLE ConvTran (χωρίς ensemble, 10 κλάσεις)')
    model = load_convtran('outputs/models/convtran_single_full.pt')
    data = np.load('data/windows/full/test.npz')
    y_pred = predict_convtran(model, data['X'])
    acc = accuracy_score(data['y'], y_pred)
    results['Single ConvTran'] = acc
    print(f'\nTest accuracy = {acc:.4f}')
    print(classification_report(data['y'], y_pred, target_names=ALL_CLASS_NAMES, zero_division=0))

    # ---- 3. Ensemble: ConvTran embeddings + Random Forest meta-classifier ----
    print_section('3. ENSEMBLE (ConvTran embeddings + Random Forest)')
    meta_clf = joblib.load('outputs/models/meta_classifier.joblib')
    emb_data = np.load('outputs/embeddings/test.npz')
    y_pred = meta_clf.predict(emb_data['X'])
    acc = accuracy_score(emb_data['y'], y_pred)
    results['Ensemble (ConvTran + RF)'] = acc
    print(f'\nTest accuracy = {acc:.4f}')
    print(classification_report(emb_data['y'], y_pred, target_names=ALL_CLASS_NAMES, zero_division=0))

    # ---- 4. Random Forest baseline (raw στατιστικά, χωρίς κανένα ConvTran) ----
    print_section('4. RANDOM FOREST BASELINE (χωρίς ConvTran)')
    rf_baseline = joblib.load('outputs/models/rf_baseline.joblib')
    raw_data = np.load('data/windows/full/test.npz')
    X_summarized = summarize_windows(raw_data['X'])
    y_pred = rf_baseline.predict(X_summarized)
    acc = accuracy_score(raw_data['y'], y_pred)
    results['RF baseline'] = acc
    print(f'\nTest accuracy = {acc:.4f}')
    print(classification_report(raw_data['y'], y_pred, target_names=ALL_CLASS_NAMES, zero_division=0))

    # ---- Συγκεντρωτικός πίνακας ----
    print_section('ΣΥΓΚΕΝΤΡΩΤΙΚΟΣ ΠΙΝΑΚΑΣ')
    for name, acc in results.items():
        print(f'{name:35s} : {acc:.4f} ({acc*100:.2f}%)')