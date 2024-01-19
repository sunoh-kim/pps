import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentivePooling, self).__init__()

        self.att_n = 1 
        self.feat_dim = hidden_dim 
        self.att_hid_dim = hidden_dim 

        self.feat2att = nn.Linear(self.feat_dim, self.att_hid_dim, bias=False)
        self.to_alpha = nn.Linear(self.att_hid_dim, self.att_n, bias=False)

        edim = hidden_dim 
        self.fc = nn.Linear(self.feat_dim, edim)

    def forward(self, feats, f_masks=None):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            feats: features where attention weights are computed; [B, A, D]
            f_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert f_masks is None or len(f_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # embedding feature vectors
        attn_f = self.feat2att(feats)  

        # compute attention weights
        dot = torch.tanh(attn_f)       
        alpha = self.to_alpha(dot)     
        if f_masks is not None:
            alpha = alpha.masked_fill(f_masks.float().unsqueeze(2).eq(0), -1e9)
        attw = F.softmax(alpha.transpose(1,2), dim=2) 

        att_feats = attw @ feats 
        if self.att_n == 1:
            att_feats = att_feats.squeeze(1)
            attw = attw.squeeze(1)

        att_feats = self.fc(att_feats)

        return att_feats, attw