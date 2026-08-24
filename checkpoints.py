import torch

CHECKPOINTS = {
    'cleaned_1': 'outputs/models/convtran_cleaned_1.pt',
    'cleaned_2': 'outputs/models/convtran_cleaned_2.pt',
    'cleaned_3': 'outputs/models/convtran_cleaned_3.pt',
}

for name, path in CHECKPOINTS.items():
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    print(f"{name}: καλύτερο epoch = {ckpt['epoch']}, val_loss = {ckpt['val_loss']:.4f}, val_acc = {ckpt['val_acc']:.4f},")