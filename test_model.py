import torch
from models.model import ConvTran

# Ίδιες διαστάσεις με τα πραγματικά σου δεδομένα (windowing.py):
# X shape ήταν (N, 60, 30) = (batch, seq_len=60, num_features=30)
# Το μοντέλο όμως θέλει (batch, num_features, seq_len) -- γι' αυτό το transpose
batch_size = 4
num_features = 30
seq_len = 60
num_classes = 3  # π.χ. cleaned_1 (A, B, C)

# Dummy δεδομένα, ήδη στο σωστό σχήμα (batch, features, seq_len)
dummy_x = torch.randn(batch_size, num_features, seq_len)

model = ConvTran(
    num_features=num_features,
    seq_len=seq_len,
    num_classes=num_classes,
    emb_size=16,
    num_heads=8,
    dim_ff=256,
    dropout=0.01
)

# Test 1: κανονική πρόβλεψη (classification)
output = model(dummy_x)
print("Classification output shape:", output.shape)
# Περιμένουμε: (4, 3)  -> 4 δείγματα, 3 κλάσεις

# Test 2: embedding (αυτό θα χρειαστείς αργότερα για το ensemble)
embedding = model(dummy_x, return_embedding=True)
print("Embedding shape:", embedding.shape)
# Περιμένουμε: (4, 16)  -> 4 δείγματα, 16 διαστάσεις (όσο το emb_size)

print("\nΑριθμός εκπαιδεύσιμων παραμέτρων:", sum(p.numel() for p in model.parameters() if p.requires_grad))