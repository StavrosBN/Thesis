"""
Προσαρμοσμένο από το επίσημο repo του ConvTran (Foumani et al., 2023).
Αρχιτεκτονική: convolutional embedding (temporal + spatial conv) -> tAPE
-> multi-head attention με eRPE -> feed-forward -> Global Average Pooling
-> γραμμικός classifier.
"""
import torch
from torch import nn
from models.positional_encoding import tAPE
from models.attention import Attention_Rel_Scl


class ConvTran(nn.Module):
    def __init__(self, num_features, seq_len, num_classes,
                 emb_size=16, num_heads=8, dim_ff=256, dropout=0.01):
        super().__init__()

        # ---- 1. Convolutional embedding (2 στάδια, "inverted bottleneck") ----
        # Στάδιο Α: temporal convolution -- σαρώνει κατά μήκος του χρόνου (kernel=[1,8])
        # για κάθε feature ξεχωριστά, εξάγοντας τοπικά χρονικά μοτίβα.
        self.embed_layer = nn.Sequential(
            nn.Conv2d(1, emb_size * 4, kernel_size=[1, 8], padding='same'),
            nn.BatchNorm2d(emb_size * 4),
            nn.GELU()
        )
        # Στάδιο Β: spatial convolution -- σαρώνει κατά μήκος όλων των features
        # (kernel=[num_features,1]) για να συνδυάσει πληροφορία ανάμεσα σε
        # διαφορετικούς αισθητήρες σε κάθε χρονική στιγμή.
        self.embed_layer2 = nn.Sequential(
            nn.Conv2d(emb_size * 4, emb_size, kernel_size=[num_features, 1], padding='valid'),
            nn.BatchNorm2d(emb_size),
            nn.GELU()
        )

        # ---- 2. Position encoding (tAPE) ----
        self.pos_encode = tAPE(emb_size, dropout=dropout, max_len=seq_len)

        # ---- 3. Multi-head attention με eRPE ----
        self.attention_layer = Attention_Rel_Scl(emb_size, num_heads, seq_len, dropout)

        self.norm1 = nn.LayerNorm(emb_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(emb_size, eps=1e-5)

        # ---- 4. Feed-forward block ----
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, emb_size),
            nn.Dropout(dropout)
        )

        # ---- 5. Global Average Pooling + τελικός classifier ----
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(emb_size, num_classes)

    def forward(self, x, return_embedding=False):
        """
        x: (batch, num_features, seq_len)  <-- ΠΡΟΣΟΧΗ σε αυτό το σχήμα, δες σημείωση παρακάτω
        return_embedding: αν True, επιστρέφει το embedding (πριν τον classifier)
                          αντί για την τελική πρόβλεψη -- αυτό θα χρειαστείς
                          αργότερα για το ensemble.
        """
        x = x.unsqueeze(1)                      # (batch, 1, num_features, seq_len)
        x = self.embed_layer(x)                 # (batch, emb_size*4, num_features, seq_len)
        x = self.embed_layer2(x).squeeze(2)     # (batch, emb_size, seq_len)
        x = x.permute(0, 2, 1)                  # (batch, seq_len, emb_size)

        x_pos = self.pos_encode(x)
        att = x + self.attention_layer(x_pos)   # residual connection
        att = self.norm1(att)

        out = att + self.feed_forward(att)      # residual connection
        out = self.norm2(out)

        out = out.permute(0, 2, 1)              # (batch, emb_size, seq_len)
        embedding = self.flatten(self.gap(out)) # (batch, emb_size)  <-- ΤΟ EMBEDDING

        if return_embedding:
            return embedding

        return self.classifier(embedding)       # (batch, num_classes)