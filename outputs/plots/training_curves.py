import matplotlib.pyplot as plt

epochs = list(range(1, 70))

train_loss = [0.9715, 0.5455, 0.4011, 0.3945, 0.3420, 0.3283, 0.2636, 0.2644, 0.2532, 0.2358,
              0.2306, 0.1997, 0.1863, 0.1989, 0.2051, 0.1944, 0.1620, 0.1846, 0.1401, 0.1562,
              0.1647, 0.1249, 0.1638, 0.1705, 0.1087, 0.1332, 0.1905, 0.1194, 0.0798, 0.1083,
              0.1088, 0.0910, 0.1357, 0.0988, 0.0858, 0.1002, 0.1181, 0.0745, 0.1157, 0.0745,
              0.0722, 0.0751, 0.0916, 0.1458, 0.0897, 0.0898, 0.0669, 0.0536, 0.0548, 0.0705,
              0.0732, 0.0814, 0.0712, 0.0440, 0.0711, 0.0714, 0.0317, 0.0558, 0.0700, 0.0831,
              0.0559, 0.0474, 0.0749, 0.0307, 0.0626, 0.0430, 0.0711, 0.0706, 0.0571]

val_loss = [0.7444, 0.9389, 0.6439, 0.3693, 0.4679, 0.2375, 0.3516, 0.9298, 0.2714, 0.3282,
            0.8843, 0.3492, 0.2234, 0.5939, 0.5047, 0.4133, 0.3808, 0.7485, 1.0424, 0.2245,
            0.3667, 0.1224, 0.2018, 0.1310, 0.8135, 0.3823, 0.4906, 0.0962, 0.2515, 0.1357,
            0.2078, 0.4456, 0.0597, 0.5259, 1.0919, 0.2558, 0.6120, 0.3305, 0.1805, 0.1644,
            0.8936, 0.2986, 0.5413, 0.0358, 0.4025, 0.1191, 0.2318, 0.0551, 2.2254, 0.2071,
            0.7690, 0.3658, 0.6838, 0.1382, 1.6205, 0.3020, 2.9124, 0.3718, 2.7845, 0.0683,
            0.1269, 0.2208, 0.6512, 0.1577, 0.2556, 0.1683, 1.0081, 0.5733, 2.1557]

BEST_EPOCH = 44
OUTPUT_PATH = "outputs/plots/figure_5_1_training_curves.png"

if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(epochs, train_loss, color="#2f6f4f", linewidth=1.8, label="Training loss")
    ax.plot(epochs, val_loss, color="#c96a2e", linewidth=1.2, alpha=0.85, label="Validation loss")
    ax.axvline(BEST_EPOCH, color="#555555", linestyle="--", linewidth=1.2)
    ax.text(BEST_EPOCH + 0.8, 1.55, f"Καλύτερο epoch ({BEST_EPOCH})",
            fontsize=9, color="#333333")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Καμπύλες εκπαίδευσης ConvTran (cleaned_1) — παράδειγμα αστάθειας validation loss")
    ax.legend(loc="upper left", frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 3.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches='tight', facecolor='white')
    print(f"Αποθηκεύτηκε: {OUTPUT_PATH}")