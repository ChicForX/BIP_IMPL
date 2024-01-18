import torch
import torch.nn as nn


# based on abs of weights
def heuristic_pruning(model, threshold):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weights = torch.abs(module.weight.data)
            mask = weights > threshold
            module.activate_flag.data = mask.float()
    return model


# pruning progressively
def iterative_pruning(model, prune_rate):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weights = torch.abs(module.weight.data)
            # already pruned in previous iterations
            already_pruned = (module.activate_flag.data == 0)
            weights[already_pruned] = 0

            threshold = torch.quantile(weights, prune_rate)
            mask = weights > threshold
            module.activate_flag.data = mask.float()

    return model


# based on rate
def onetime_pruning_by_rate(model, prune_rate):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weights = torch.abs(module.weight.data)
            threshold = torch.quantile(weights, prune_rate)
            mask = weights > threshold
            module.activate_flag.data = mask.float()
    return model


# freeze vars by name
def freeze_vars_by_key(model, key):
    for i, v in model.named_modules():
        if hasattr(v, key):
            if getattr(v, key) is not None:
                getattr(v, key).requires_grad = False
    return model


# unfreeze vars by name
def unfreeze_vars_by_key(model, key):
    for i, v in model.named_modules():
        if hasattr(v, key):
            if getattr(v, key) is not None:
                getattr(v, key).requires_grad = True
    return model
