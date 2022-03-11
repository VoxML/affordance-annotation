import torch
import torch.nn as nn

class AbsoluteOrientationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AbsoluteOrientationNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


class AbsoluteOrientationAndObjNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, emb_size, num_objs):
        super(AbsoluteOrientationAndObjNet, self).__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(num_objs, emb_size)
        self.l1 = nn.Linear(input_size + emb_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, orientation, obj_lab):
        obj_emb = self.embedding(obj_lab)
        concat = torch.cat([orientation, obj_emb], dim=-1)
        out = self.l1(concat)
        out = self.relu(out)
        out = self.l2(out)
        return out