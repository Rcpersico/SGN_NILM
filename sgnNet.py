import torch
import torch.nn as nn
import torch.nn.functional as F



#Blocks
class ConvBlock(nn.Module):
    def __init__(self, c, k = 3, p = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(c, c, k, padding = k//2)
        self.conv2 = nn.Conv1d(c, c, k, padding = k//2)
        self.bn1 = nn.BatchNorm1d(c)
        self.bn2 = nn.BatchNorm1d(c)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.dropout(y)
        y = F.relu(self.bn2(self.conv2(y)))
        return x + y
    



class TCNBlock(nn.Module):
    def __init__(self, c, k = 3, d = 1, p=0.1):
        super().__init__()
        self.pad = (d*(k-1), 0)
        self.conv1 = nn.Conv1d(c, c, k , dilation=d)
        self.conv2 = nn.Conv1d(c, c, k , dilation=d)
        self.bn1 = nn.BatchNorm1d(c)
        self.bn2 = nn.BatchNorm1d(c)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        y = F.pad(x, self.pad)
        y = F.relu(self.bn1(self.conv1(y)))
        y = self.dropout(y)
        y = F.pad(y, self.pad)
        y = F.relu(self.bn2(self.conv2(y)))
        return x + y
    




#Backbone Class

class Backbone(nn.Module):
    def __init__(self, in_ch=1 , hid=64, depth = 6, kind = "tcn", k = 3, p = 0.1, 
                 dilations = (1,2,4,8,16,32)):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, hid, kernel_size = 3, padding = 1)
        blocks = []
        if kind == "tcn":
            if dilations is None:
                dilations = tuple(2**i for i in range(depth))
            for d in dilations:
                blocks.append(TCNBlock(hid, k=k, d=d, p=p))

        elif kind == "cnn":
            for _ in range(depth):
                blocks.append(ConvBlock(hid, k=k, p=p ))

        else:
            raise ValueError(f"Unknown kind: {kind}")
        
        self.blocks = nn.ModuleList(blocks)


    def forward(self , x):
        h = F.relu(self.stem(x))
        for b in self.blocks:
            h = b(h)
        return h
    



#SGN Head
class SGN(nn.Module):
    def __init__(self, in_ch = 1, hid = 64, kind = "tcn", k =3, p = 0.2,
                 dilations = (1,2,4,8,16,32), out_len = 1):
        super().__init__()
        self.out_len = out_len
        self.backbone = Backbone(in_ch = in_ch, hid = hid, depth = len(dilations) if kind =="tcn" else 6,
                                 kind = kind, k=k,p=p, dilations=dilations if kind == "tcn" else None)
        self.head_reg = nn.Conv1d(hid, 1, kernel_size = 1)
        self.head_cls = nn.Conv1d(hid, 1, kernel_size = 1)

        #if you want seq2subseq you need to increase out_len from 1
        self.pool = nn.AdaptiveAvgPool1d(out_len)


    def forward(self, x):
        h = self.backbone(x)
        h = self.pool(h)
        reg = self.head_reg(h).squeeze(1)
        cls_logits = self.head_cls(h).squeeze(1)  
        cls_prob = torch.sigmoid(cls_logits)      # probs for gating
        power = reg * cls_prob                    # SGN gating
        return power, cls_logits, reg








