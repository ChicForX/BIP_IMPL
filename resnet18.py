import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18()

        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # fine-grained pruning: same shape as weights
                module.register_buffer('activate_flag', torch.ones_like(module.weight.data))

    def forward(self, x):
        for name, module in self.model.named_children():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                module.weight.data *= module.activate_flag
            x = module(x)

        return x