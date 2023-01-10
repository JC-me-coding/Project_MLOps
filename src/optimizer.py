import torch
def make_optimizer(optimizer_name, model, config):
    # ToDo: Use more exotic ones from timm / parametrize
    optimizer = None
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    # elif ...
    return optimizer