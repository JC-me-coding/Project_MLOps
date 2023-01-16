import torch
def make_optimizer(optimizer_name, model, config):
    # ToDo: Use more exotic ones from timm / parametrize
    optimizer = None
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.hyperparameters.learning_rate,
                                     weight_decay=config.hyperparameters.weight_decay)
    elif optimizer_name == "adamW":
        optimizer = torch.optim.AdamW(model.parameters(),
                                lr=config.hyperparameters.learning_rate,
                                weight_decay=config.hyperparameters.weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.hyperparameters.learning_rate,
                                    weight_decay=config.hyperparameters.weight_decay)

    return optimizer