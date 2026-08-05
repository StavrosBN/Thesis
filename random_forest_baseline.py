import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FULL_WINDOWS_DIR = 'data/windows/full'
OUTPUT_MODELS_DIR = 'outputs/models'
OUTPUT_PLOTS_DIR = 'outputs/plots'
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']


def load_windows(split_name):
    data = np.load(os.path.join(FULL_WINDOWS_DIR, f'{split_name}.npz'))
    return data['X'], data['y']  # X: (N, 60, 30)


def summarize_windows(X):
    """
    Μετατρέπει κάθε παράθυρο (60 timesteps x 30 features) σε ένα tabular
    διάνυσμα χαρακτηριστικών, υπολογίζοντας μέσο όρο και τυπική απόκλιση
    ανά feature μέσα στο παράθυρο -- έτσι το Random Forest βλέπει ακριβώς
    το ίδιο 'χρονικό απόσπασμα' με το ConvTran, απλά συμπυκνωμένο με απλά
    στατιστικά αντί για μαθημένα embeddings.
    """
    mean_features = X.mean(axis=1)  # (N, 30)
    std_features = X.std(axis=1)    # (N, 30)
    return np.concatenate([mean_features, std_features], axis=1)  # (N, 60)


if __name__ == '__main__':
    # ---- Φόρτωση ΤΩΝ ΙΔΙΩΝ windows με το ensemble, όχι νέο split ----
    X_train_raw, y_train = load_windows('meta_train')
    X_test_raw, y_test = load_windows('test')

    X_train = summarize_windows(X_train_raw)
    X_test = summarize_windows(X_test_raw)
    print(f'meta_train (summarized): X={X_train.shape}, y={y_train.shape}')
    print(f'test (summarized):       X={X_test.shape}, y={y_test.shape}')

    # ---- Ίδιος τύπος/παράμετροι Random Forest με τον meta-classifier ----
    rf_baseline = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    rf_baseline.fit(X_train, y_train)

    y_pred = rf_baseline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'\n=== Random Forest Baseline (raw στατιστικά, χωρίς ConvTran) ===')
    print(f'Test accuracy: {acc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Πρόβλεψη')
    plt.ylabel('Πραγματική κλάση')
    plt.title(f'Random Forest Baseline Confusion Matrix (test accuracy={acc:.4f})')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, 'rf_baseline_confusion_matrix.png'), dpi=150)
    print(f'\nΑποθηκεύτηκε: {OUTPUT_PLOTS_DIR}/rf_baseline_confusion_matrix.png')

    joblib.dump(rf_baseline, os.path.join(OUTPUT_MODELS_DIR, 'rf_baseline.joblib'))
    print(f'Αποθηκεύτηκε: {OUTPUT_MODELS_DIR}/rf_baseline.joblib')