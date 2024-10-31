import torch
import torch.nn as nn
from transformers import DebertaV2Model
from torchvision import models

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.deberta = DebertaV2Model.from_pretrained(TEXT_MODEL_NAME)
        self.proj = nn.Linear(self.deberta.config.hidden_size, 512)

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.proj(outputs[:, 0, :])

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        densenet = models.densenet121(pretrained=True)
        self.features = densenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(1024, 512)

    def forward(self, images):
        features = self.features(images)
        pooled = self.avgpool(features).view(features.size(0), -1)
        return self.proj(pooled)
