import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

EMBEDDINGS_DIR = 'outputs/embeddings'
OUTPUT_MODELS_DIR = 'outputs/models'
OUTPUT_PLOTS_DIR = 'outputs/plots'
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # σειρά = 0..9, όπως το LabelEncoder


def load_embeddings(split_name):
    data = np.load(os.path.join(EMBEDDINGS_DIR, f'{split_name}.npz'))
    return data['X'], data['y']


if __name__ == '__main__':
    # ---- Φόρτωση των embeddings (ΟΧΙ raw δεδομένων -- ήδη συμπυκνωμένα από τα 3 ConvTran) ----
    X_train, y_train = load_embeddings('meta_train')
    X_test, y_test = load_embeddings('test')
    print(f'meta_train: X={X_train.shape}, y={y_train.shape}')
    print(f'test:       X={X_test.shape}, y={y_test.shape}')

    # ---- Εκπαίδευση του meta-classifier ----
    # class_weight='balanced' αντισταθμίζει τις μικρές διαφορές στο μέγεθος ανά κλάση
    meta_clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    meta_clf.fit(X_train, y_train)

    # ---- Πρόβλεψη & αξιολόγηση στο ΑΓΝΩΣΤΟ test set ----
    y_pred = meta_clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'\n=== Ensemble (ConvTran embeddings + Random Forest) ===')
    print(f'Test accuracy: {acc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # ---- Confusion matrix ----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Πρόβλεψη')
    plt.ylabel('Πραγματική κλάση')
    plt.title(f'Ensemble Confusion Matrix (test accuracy={acc:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'ensemble_confusion_matrix.png'), dpi=150)
    print(f'\nΑποθηκεύτηκε: {OUTPUT_PLOTS_DIR}/ensemble_confusion_matrix.png')

    # ---- Αποθήκευση του εκπαιδευμένου meta-classifier ----
    joblib.dump(meta_clf, os.path.join(OUTPUT_MODELS_DIR, 'meta_classifier.joblib'))
    print(f'Αποθηκεύτηκε: {OUTPUT_MODELS_DIR}/meta_classifier.joblib')