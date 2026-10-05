import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

warnings.filterwarnings("ignore")

OUT = "presentation_figures"
os.makedirs(OUT, exist_ok=True)

CLASS_NAMES = list("ABCDEFGHIJ")
GROUP_COLORS = {"ABC": "#4C78A8", "DEF": "#F58518", "GHIJ": "#54A24B"}
DRIVER_COLORS = ["#4C78A8"] * 3 + ["#F58518"] * 3 + ["#54A24B"] * 4

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Αποθηκεύτηκε:", path)


# ---------------------------------------------------------------- helpers
def box(ax, x, y, w, h, text, color="#DDEAF7", edge="#335577", fs=11, bold=False):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       fc=color, ec=edge, lw=1.5)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=18, lw=1.8, color="#333333"))


# ---------------------------------------------------------------- fig01
def fig_class_distribution():
    df = pd.read_csv("data/drivers_data.csv")
    counts = df["Class"].astype(str).str.strip().str.upper().value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(CLASS_NAMES, counts.values, color=DRIVER_COLORS)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 150, f"{v:,}", ha="center", fontsize=10)
    ax.set_xlabel("Οδηγός")
    ax.set_ylabel("Αριθμός δειγμάτων (1 δείγμα = 1 sec)")
    ax.set_title(f"Κατανομή δεδομένων ανά οδηγό (σύνολο {counts.sum():,})")
    # υπόμνημα ομάδων
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=GROUP_COLORS["ABC"], label="Ομάδα 1 (A-C)"),
                       Patch(color=GROUP_COLORS["DEF"], label="Ομάδα 2 (D-F)"),
                       Patch(color=GROUP_COLORS["GHIJ"], label="Ομάδα 3 (G-J)")],
              frameon=False, loc="upper right")
    save(fig, "fig01_class_distribution.png")


# ---------------------------------------------------------------- fig02
def fig_mutual_information():
    from sklearn.feature_selection import mutual_info_classif
    df = pd.read_csv("data/drivers_data.csv")
    # διορθώνουμε διπλά ονόματα στηλών όπως στο notebook
    cols = pd.Series(df.columns)
    for dupe in cols[cols.duplicated()].unique():
        cols[cols == dupe] = [f"{dupe}_{i}" if i != 0 else dupe for i in range(sum(cols == dupe))]
    df.columns = cols
    X = df.drop(columns=["Class", "PathOrder", "Time(s)"], errors="ignore")
    y = df["Class"].astype(str).str.strip().str.upper()
    print("Υπολογισμός Mutual Information (μπορεί να πάρει λίγο)...")
    mi = pd.Series(mutual_info_classif(X, y, random_state=42), index=X.columns)
    top = mi.sort_values(ascending=False).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top.index, top.values, color="#4C78A8")
    ax.set_xlabel("Mutual Information με την κλάση-οδηγό")
    ax.set_title("Τα 12 πιο πληροφοριακά σήματα (Mutual Information)")
    save(fig, "fig02_mutual_information.png")


# ---------------------------------------------------------------- fig03
def fig_iqr_clipping():
    df = pd.read_csv("data/drivers_data.csv")
    feature = "Engine_speed"
    q1, q3 = df[feature].quantile(0.25), df[feature].quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    clipped = df[feature].clip(lo, hi)
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    axes[0].boxplot(df[feature].dropna(), patch_artist=True, boxprops=dict(facecolor="#F4C7A1"))
    axes[0].set_title("Πριν το IQR clipping")
    axes[1].boxplot(clipped.dropna(), patch_artist=True, boxprops=dict(facecolor="#A9D3A0"))
    axes[1].set_title("Μετά το IQR clipping")
    axes[0].set_ylabel(feature)
    for a in axes:
        a.set_xticks([])
    fig.suptitle("Επίδραση του IQR clipping (όρια Q1 - 1.5·IQR, Q3 + 1.5·IQR)", fontsize=13)
    save(fig, "fig03_iqr_clipping.png")


# ---------------------------------------------------------------- fig04
def fig_split_and_windowing():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.5), gridspec_kw={"height_ratios": [1, 1.3]})

    # --- πάνω: χρονολογικός χωρισμός μιας διαδρομής (block)
    parts = [("ConvTran train\n70%", 0.70, "#4C78A8"),
             ("Val\n10%", 0.10, "#F2B134"),
             ("Meta-train\n10%", 0.10, "#54A24B"),
             ("Test\n10%", 0.10, "#D1495B")]
    x = 0
    for label, w, c in parts:
        ax1.add_patch(Rectangle((x, 0), w, 1, fc=c, ec="white", lw=2))
        ax1.text(x + w / 2, 0.5, label, ha="center", va="center", color="white",
                 fontsize=11, fontweight="bold")
        x += w
    ax1.annotate("", xy=(1.0, -0.15), xytext=(0, -0.15),
                 arrowprops=dict(arrowstyle="-|>", lw=1.6))
    ax1.text(0.5, -0.45, "χρόνος μέσα σε μία διαδρομή (το σχήμα επαναλαμβάνεται σε ΚΑΘΕ διαδρομή κάθε οδηγού)",
             ha="center", fontsize=10)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.7, 1.1)
    ax1.axis("off")
    ax1.set_title("Χρονολογικός χωρισμός ανά διαδρομή (χωρίς ανακάτεμα)", fontsize=13)

    # --- κάτω: sliding windows
    L, S, n = 60, 15, 180
    ax2.add_patch(Rectangle((0, 0), n, 1, fc="#EEEEEE", ec="#555555"))
    cols = ["#4C78A8", "#F58518", "#54A24B", "#D1495B"]
    for i, start in enumerate(range(0, 4 * S, S)):
        ax2.add_patch(Rectangle((start, 1.15 + i * 0.35), L, 0.28, fc=cols[i], alpha=0.85, ec="white"))
        ax2.text(start + L / 2, 1.29 + i * 0.35, f"παράθυρο {i + 1}", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")
    ax2.text(n / 2, 0.5, "σήματα CAN-bus (1 δείγμα/sec)", ha="center", va="center")
    ax2.annotate("", xy=(L, 0.95), xytext=(0, 0.95), arrowprops=dict(arrowstyle="<->"))
    ax2.text(L / 2, 0.8, "L = 60 sec", ha="center", va="top", fontsize=10)
    ax2.annotate("", xy=(S, -0.15), xytext=(0, -0.15), arrowprops=dict(arrowstyle="<->"))
    ax2.text(S / 2, -0.3, "stride = 15 sec\n(75% επικάλυψη)", ha="center", va="top", fontsize=10)
    ax2.set_xlim(-2, n + 2)
    ax2.set_ylim(-1.0, 2.7)
    ax2.axis("off")
    ax2.set_title("Sliding windows: 60 × 30 (δευτερόλεπτα × σήματα) ανά δείγμα", fontsize=13)
    fig.tight_layout()
    save(fig, "fig04_split_and_windowing.png")


# ---------------------------------------------------------------- fig05
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    box(ax, 0.1, 2.7, 1.9, 1.1, "Ακατέργαστα\nδεδομένα\nCAN-bus", "#EEEEEE", fs=10)
    box(ax, 2.5, 2.7, 1.9, 1.1, "Καθαρισμός\n+ 30 σήματα\n+ IQR clipping", "#EEEEEE", fs=10)
    box(ax, 4.9, 2.7, 1.9, 1.1, "Split + Windows\n60 × 30", "#EEEEEE", fs=10)
    arrow(ax, 2.0, 3.25, 2.5, 3.25)
    arrow(ax, 4.4, 3.25, 4.9, 3.25)

    # 3 ConvTran
    ys = [4.9, 2.9, 0.9]
    names = [("ConvTran #1", "Οδηγοί A, B, C", "ABC"),
             ("ConvTran #2", "Οδηγοί D, E, F", "DEF"),
             ("ConvTran #3", "Οδηγοί G, H, I, J", "GHIJ")]
    for y, (t, sub, g) in zip(ys, names):
        box(ax, 7.4, y, 2.1, 1.0, f"{t}\n{sub}", GROUP_COLORS[g], edge="#222222", fs=10, bold=True)
        arrow(ax, 6.8, 3.25, 7.4, y + 0.5)
        arrow(ax, 9.5, y + 0.5, 10.1, 3.25)
        ax.text(9.8, y + 0.85, "16-d", fontsize=9, ha="center")

    box(ax, 10.1, 2.4, 1.3, 1.7, "Concat\nembeddings\n(48-d)", "#FFF2CC", fs=10)
    box(ax, 11.7, 2.4, 1.6, 1.7, "Random\nForest\n(200 δέντρα)", "#F8CBAD", fs=10, bold=True)
    arrow(ax, 11.4, 3.25, 11.7, 3.25)
    ax.text(12.5, 2.1, "Πρόβλεψη:\nοδηγός A-J", ha="center", va="top", fontsize=10, fontweight="bold")

    ax.text(8.45, 6.2, "Εκπαίδευση: 70% (train) + 10% (val)", ha="center", fontsize=9, style="italic")
    ax.text(11.4, 4.5, "Εκπαίδευση: 10% (meta-train)", ha="center", fontsize=9, style="italic")
    ax.text(6.5, 0.15, "Τελική αξιολόγηση: κοινό test set (τελευταίο 10% κάθε διαδρομής)",
            ha="center", fontsize=10, fontweight="bold", color="#D1495B")
    save(fig, "fig05_pipeline.png")


# ---------------------------------------------------------------- fig06
def fig_convtran_architecture():
    fig, ax = plt.subplots(figsize=(7.5, 11))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)
    ax.axis("off")

    items = [
        ("Είσοδος: 30 σήματα × 60 sec", "#EEEEEE", ""),
        ("Temporal Conv (kernel 1×8)\n+ BatchNorm + GELU", "#DDEAF7", "→ 64 κανάλια"),
        ("Spatial Conv (kernel 30×1)\n+ BatchNorm + GELU", "#DDEAF7", "→ 16 κανάλια × 60 βήματα"),
        ("tAPE\n(absolute position encoding)", "#FFF2CC", ""),
        ("Multi-Head Attention (8 heads)\n+ eRPE (relative position)", "#F8CBAD", ""),
        ("Add & LayerNorm  (residual)", "#EEEEEE", ""),
        ("Feed-Forward (16 → 256 → 16)", "#E2F0D9", ""),
        ("Add & LayerNorm  (residual)", "#EEEEEE", ""),
        ("Global Average Pooling", "#DDEAF7", ""),
        ("EMBEDDING (16-d)", "#C6E0B4", "→ χρησιμοποιείται στο ensemble"),
        ("Γραμμικός classifier", "#EEEEEE", "→ πιθανότητες οδηγών"),
    ]
    h, gap = 1.35, 0.6
    y = 22 - h - 0.2
    ys = []
    for text, color, note in items:
        box(ax, 1.2, y, 5.6, h, text, color, fs=10.5, bold=text.startswith("EMBEDDING"))
        if note:
            ax.text(7.0, y + h / 2, note, va="center", fontsize=9.5, style="italic")
        ys.append(y)
        y -= h + gap
    for i in range(len(ys) - 1):
        arrow(ax, 4.0, ys[i], 4.0, ys[i + 1] + h)
    save(fig, "fig06_convtran_architecture.png")


# ---------------------------------------------------------------- αξιολόγηση
def compute_predictions():
    """Υπολογίζει προβλέψεις και των 4 προσεγγίσεων από τα αποθηκευμένα μοντέλα."""
    import torch
    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    from models.model import ConvTran

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_convtran(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        m = ConvTran(num_features=ck["num_features"], seq_len=ck["seq_len"],
                     num_classes=ck["num_classes"], emb_size=ck["emb_size"],
                     num_heads=ck["num_heads"], dim_ff=ck["dim_ff"], dropout=ck["dropout"]).to(device)
        m.load_state_dict(ck["model_state_dict"])
        m.eval()
        return m

    test = np.load("data/windows/full/test.npz")
    X, y = test["X"], test["y"]

    # Single ConvTran
    m = load_convtran("outputs/models/convtran_single_full.pt")
    with torch.no_grad():
        t = torch.from_numpy(X.astype(np.float32).transpose(0, 2, 1)).to(device)
        y_single = m(t).argmax(1).cpu().numpy()

    # RF baseline
    rf = joblib.load("outputs/models/rf_baseline.joblib")
    y_rf = rf.predict(np.concatenate([X.mean(axis=1), X.std(axis=1)], axis=1))

    # Ensemble
    meta = joblib.load("outputs/models/meta_classifier.joblib")
    emb = np.load("outputs/embeddings/test.npz")
    y_ens = meta.predict(emb["X"])

    preds = {"RF baseline": y_rf, "Single ConvTran": y_single, "Ensemble (ConvTran + RF)": y_ens}
    accs = {k: accuracy_score(y, p) for k, p in preds.items()}
    f1s = {k: f1_score(y, p, average=None, labels=range(10), zero_division=0) for k, p in preds.items()}
    macro = {k: f1_score(y, p, average="macro", zero_division=0) for k, p in preds.items()}

    # 3 επιμέρους ConvTran στα δικά τους test sets
    sub = {}
    for i, g in enumerate(["cleaned_1", "cleaned_2", "cleaned_3"], start=1):
        mm = load_convtran(f"outputs/models/convtran_{g}.pt")
        d = np.load(f"data/windows/{g}/test.npz")
        with torch.no_grad():
            tt = torch.from_numpy(d["X"].astype(np.float32).transpose(0, 2, 1)).to(device)
            pp = mm(tt).argmax(1).cpu().numpy()
        sub[g] = accuracy_score(d["y"], pp)
    return accs, f1s, macro, sub


def fig_accuracy_comparison(accs, macro, sub):
    labels = ["RF baseline\n(mean+std)", "Single\nConvTran", "Ensemble\n(ConvTran + RF)"]
    keys = ["RF baseline", "Single ConvTran", "Ensemble (ConvTran + RF)"]
    vals = [accs[k] * 100 for k in keys]
    colors = ["#F58518", "#54A24B", "#4C78A8"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.axhline(10, color="#999999", ls="--", lw=1)
    ax.text(2.45, 11.5, "τυχαία επιλογή (10%)", ha="right", fontsize=9, color="#666666")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Πρόβλημα 10 οδηγών (A-J)")

    sv = [sub["cleaned_1"] * 100, sub["cleaned_2"] * 100, sub["cleaned_3"] * 100]
    sl = ["Ομάδα 1\n(A-B-C)\n3 κλάσεις", "Ομάδα 2\n(D-E-F)\n3 κλάσεις", "Ομάδα 3\n(G-H-I-J)\n4 κλάσεις"]
    bars = ax2.bar(sl, sv, color=[GROUP_COLORS["ABC"], GROUP_COLORS["DEF"], GROUP_COLORS["GHIJ"]])
    for b, v in zip(bars, sv):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_title("Τα 3 επιμέρους ConvTran (δικό τους test)")
    fig.tight_layout()
    save(fig, "fig07_accuracy_comparison.png")


def fig_perclass_f1(f1s):
    keys = ["RF baseline", "Single ConvTran", "Ensemble (ConvTran + RF)"]
    colors = ["#F58518", "#54A24B", "#4C78A8"]
    x = np.arange(10)
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (k, c) in enumerate(zip(keys, colors)):
        ax.bar(x + (i - 1) * w, f1s[k], w, label=k, color=c)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_xlabel("Οδηγός")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.1)
    ax.set_title("F1-score ανά οδηγό")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    save(fig, "fig08_perclass_f1.png")

 
OUT = "presentation_figures"
os.makedirs(OUT, exist_ok=True)
 
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
 
DRV = list("ABCDEFGHIJ")
C_RF, C_SINGLE, C_ENS = "#F58518", "#54A24B", "#4C78A8"
G1, G2, G3 = "#4C78A8", "#F58518", "#54A24B"
 
# ---- Αριθμοί από τη διπλωματική -------------------------------------------
ACC = {"RF baseline": 55.30, "Single ConvTran": 53.95, "Ensemble": 67.24}   # Πίνακας 7.2
SUB = {"f1 (A,B,C)": 96.62, "f2 (D,E,F)": 50.76, "f3 (G,H,I,J)": 73.22}   # Πίνακας 7.2
 
F1_ENS = [1.00, 0.75, 0.47, 0.52, 0.96, 0.57, 0.25, 0.73, 0.67, 0.78]       # Πίνακας 7.3
F1_RF = [1.00, 0.55, 0.68, 0.64, 0.69, 0.33, 0.21, 0.54, 0.27, 0.53]        # Πίνακας 7.4
F1_SINGLE = [0.92, 0.79, 0.08, 0.34, 0.92, 0.38, 0.28, 0.53, 0.51, 0.62]    # Πίνακας 7.5
F1_SUB = [1.00, 0.96, 0.94, 0.47, 0.77, 0.19, 0.83, 0.74, 0.43, 0.78]       # Πίνακας 7.7
 
MACRO = {  # Πίνακας 7.6: (Precision, Recall, F1)
    "RF baseline": (0.69, 0.54, 0.54),
    "Single ConvTran": (0.57, 0.56, 0.54),
    "Ensemble": (0.68, 0.68, 0.67),
}
 
 
def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Αποθηκεύτηκε:", p)
 
 
def fig07():
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.2, 1]})
    labels = ["RF baseline\n(mean+std)", "Single\nConvTran", "Ensemble\n(3 ConvTran + RF)"]
    vals = list(ACC.values())
    bars = ax.bar(labels, vals, color=[C_RF, C_SINGLE, C_ENS])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")
    ax.axhline(10, color="#999999", ls="--", lw=1)
    ax.text(2.45, 12, "τυχαία επιλογή (10%)", ha="right", fontsize=9, color="#666666")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Κοινό test set: 519 παράθυρα, 10 οδηγοί")
 
    sl = ["f1\n(A,B,C)", "f2\n(D,E,F)", "f3\n(G,H,I,J)"]
    bars = ax2.bar(sl, list(SUB.values()), color=[G1, G2, G3])
    for b, v in zip(bars, SUB.values()):
        ax2.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")
    ax2.set_ylim(0, 100)
    ax2.set_title("Επιμέρους ConvTran (δικό τους test set)")
    fig.tight_layout()
    save(fig, "fig07_accuracy_comparison.png")
 
 
def fig08():
    x = np.arange(10)
    w = 0.27
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w, F1_RF, w, label="RF baseline", color=C_RF)
    ax.bar(x, F1_SINGLE, w, label="Single ConvTran", color=C_SINGLE)
    ax.bar(x + w, F1_ENS, w, label="Ensemble", color=C_ENS)
    ax.set_xticks(x)
    ax.set_xticklabels(DRV)
    ax.set_xlabel("Οδηγός")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.15)
    ax.set_title("F1-score ανά οδηγό (κοινό test set)")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    save(fig, "fig08_perclass_f1.png")
 
 
def fig09():
    x = np.arange(10)
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w / 2, F1_SUB, w, label="στο δικό του υπο-πρόβλημα (3-4 κλάσεις)", color="#BBBBBB")
    ax.bar(x + w / 2, F1_ENS, w, label="στο πλήρες Ensemble (10 κλάσεις)", color=C_ENS)
    for i in (5, 6):  # F και G
        d = F1_ENS[i] - F1_SUB[i]
        ax.annotate(f"{d:+.2f}", xy=(i, max(F1_SUB[i], F1_ENS[i]) + 0.04), ha="center",
                    fontweight="bold", color="#D1495B")
    ax.set_xticks(x)
    ax.set_xticklabels(DRV)
    ax.set_xlabel("Οδηγός")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1.2)
    ax.set_title("Από το υπο-πρόβλημα στο πλήρες Ensemble: ο F ανεβαίνει, ο G πέφτει")
    ax.legend(frameon=False, loc="upper center", ncol=2)
    save(fig, "fig09_subproblem_vs_ensemble.png")
 
 
def fig10():
    names = list(MACRO.keys())
    metrics = ["Macro Precision", "Macro Recall", "Macro F1"]
    x = np.arange(3)
    w = 0.27
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (n, c) in enumerate(zip(names, [C_RF, C_SINGLE, C_ENS])):
        vals = MACRO[n]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=n, color=c)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 0.9)
    ax.set_title("Macro μετρικές (κοινό test set)")
    ax.legend(frameon=False, ncol=3, loc="upper center")
    save(fig, "fig10_macro_metrics.png")
 
 
if __name__ == "__main__":
    fig_class_distribution()
    fig_mutual_information()
    fig_iqr_clipping()
    fig_split_and_windowing()
    fig_pipeline()
    fig_convtran_architecture()
    fig07()
    fig08()
    fig09()
    fig10()
    print("Τέλος. Εικόνες στον φάκελο:", OUT)



'''if __name__ == "__main__":
    fig_class_distribution()
    fig_mutual_information()
    fig_iqr_clipping()
    fig_split_and_windowing()
    fig_pipeline()
    fig_convtran_architecture()

    accs, f1s, macro, sub = compute_predictions()
    print("\nAccuracy:", {k: round(v, 4) for k, v in accs.items()})
    print("Sub-models:", {k: round(v, 4) for k, v in sub.items()})
    fig_accuracy_comparison(accs, macro, sub)
    fig_perclass_f1(f1s)
    print("\nΤέλος. Οι εικόνες είναι στον φάκελο:", OUT)
    '''