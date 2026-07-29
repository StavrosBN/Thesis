import pandas as pd
import numpy as np
import os

L = 60        # μήκος παραθύρου σε δευτερόλεπτα (1Hz -> 60 γραμμές)
STRIDE = 15   # βήμα μετακίνησης (50% επικάλυψη)

# Στήλες που ΔΕΝ είναι features (μεταδεδομένα, όχι input στο μοντέλο)
NON_FEATURE_COLS = ['Class', 'PathOrder', 'block_id', 'split']


def make_windows(group_df, feature_cols, L, stride):
    """
    Κόβει sliding windows ΜΕΣΑ σε ένα ενιαίο, συνεχόμενο κομμάτι δεδομένων
    (π.χ. ένα συγκεκριμένο block_id + split). Ποτέ δεν διασχίζει block ή split.
    """
    values = group_df[feature_cols].values  # shape: (n_rows, n_features)
    n_rows = len(values)

    windows = []
    for start in range(0, n_rows - L + 1, stride):
        windows.append(values[start:start + L])

    return np.array(windows)  # shape: (n_windows, L, n_features)


def process_dataset(csv_path, out_dir, L=60, stride=15):
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]

    os.makedirs(out_dir, exist_ok=True)

    for split_name in ['convtran_train', 'convtran_val', 'meta_train', 'test']:
        split_df = df[df['split'] == split_name]

        X_list, y_list = [], []
        # windowing ξεχωριστά ανά block_id, ώστε να μη σμίξουν δύο trips
        for block_id, block_df in split_df.groupby('block_id'):
            windows = make_windows(block_df, feature_cols, L, stride)
            if len(windows) == 0:
                continue  # πολύ μικρό κομμάτι, δεν βγάζει ούτε ένα window
            X_list.append(windows)
            y_list.append(np.full(len(windows), block_df['Class'].iloc[0]))

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        out_path = os.path.join(out_dir, f'{split_name}.npz')
        np.savez(out_path, X=X, y=y)
        print(f'{out_path}: X={X.shape}, y={y.shape}, class counts={dict(zip(*np.unique(y, return_counts=True)))}')


# ---------- Εφαρμογή σε κάθε ομάδα ----------
groups = {
    'cleaned_1': 'data/cleaned_1_split.csv',
    'cleaned_2': 'data/cleaned_2_split.csv',
    'cleaned_3': 'data/cleaned_3_split.csv',
}

for name, path in groups.items():
    print(f'=== {name} ===')
    process_dataset(path, out_dir=f'data/windows/{name}', L=L, stride=STRIDE)
    print()


# μήκος παραθύρου σε δευτερόλεπτα (1Hz -> 60 γραμμές)
# ΣΗΜΕΙΩΣΗ ΓΙΑ ΤΗ ΔΙΠΛΩΜΑΤΙΚΗ: μειώθηκε από 30 σε 15 (75% επικάλυψη
# αντί για 50%) επειδή με stride=30 τα training sets ήταν πολύ μικρά
# (720-890 δείγματα ανά ομάδα, με κάποιες κλάσεις μόλις ~190-200 windows),
# ανεπαρκές μέγεθος για ένα transformer-based μοντέλο σαν το ConvTran.
# Το μικρότερο stride αυξάνει τον αριθμό δειγμάτων μέσω επικάλυψης
# (μια μορφή data augmentation), με το τίμημα ότι διαδοχικά παράθυρα
# μοιράζονται περισσότερες κοινές γραμμές μεταξύ τους (λιγότερη
# ανεξάρτητη πληροφορία ανά δείγμα). ΝΑ ΣΥΖΗΤΗΘΕΙ ΜΕ CLAUDE ΠΡΙΝ ΤΗ
# ΣΥΓΓΡΑΦΗ: πώς να διατυπωθεί σωστά αυτό το trade-off στο κείμενο,
# και αν χρειάζεται να αναφερθεί ως περιορισμός (limitation) της μελέτης.