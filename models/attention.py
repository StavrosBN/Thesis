"""
Προσαρμοσμένο από το επίσημο repo του ConvTran.
Υλοποιεί το eRPE (efficient Relative Position Encoding), Ενότητα 4.2 του paper.
"""
import torch
import torch.nn as nn
from einops import rearrange


class Attention_Rel_Scl(nn.Module):
    """
    Multi-head self-attention με eRPE: αντί για το κλασικό positional encoding
    πριν το attention, εδώ προστίθεται ένα learnable scalar bias (relative_bias)
    ΜΕΤΑ το softmax, βασισμένο μόνο στη σχετική απόσταση (i-j) μεταξύ δύο θέσεων
    -- ένα scalar ανά σχετική απόσταση, όχι ολόκληρο vector, γι' αυτό είναι
    πολύ πιο "ελαφρύ" σε αριθμό παραμέτρων από τα προηγούμενα relative encodings
    (Πίνακας 1 του paper).
    """

    def __init__(self, emb_size, num_heads, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.scale = emb_size ** -0.5

        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)

        # Ένα learnable βάρος για κάθε δυνατή σχετική απόσταση (2*seq_len - 1 συνολικά)
        self.relative_bias_table = nn.Parameter(torch.zeros((2 * self.seq_len - 1), num_heads))
        coords = torch.meshgrid((torch.arange(1), torch.arange(self.seq_len)))
        coords = torch.flatten(torch.stack(coords), 1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords[1] += self.seq_len - 1
        relative_coords = rearrange(relative_coords, 'c h w -> h w c')
        relative_index = relative_coords.sum(-1).flatten().unsqueeze(1)
        self.register_buffer("relative_index", relative_index)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.LayerNorm(emb_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        k = self.key(x).reshape(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        v = self.value(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        q = self.query(x).reshape(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)

        attn = torch.matmul(q, k) * self.scale
        attn = nn.functional.softmax(attn, dim=-1)  # softmax ΠΡΙΝ το relative bias

        relative_bias = self.relative_bias_table.gather(0, self.relative_index.repeat(1, self.num_heads))
        relative_bias = rearrange(relative_bias, '(h w) c -> 1 c h w', h=seq_len, w=seq_len)
        attn = attn + relative_bias  # το eRPE προστίθεται ΜΕΤΑ το softmax

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.to_out(out)
        return out
    