import torch.nn as nn
def make_loss_func(loss_func_name):
    if loss_func_name == "cross_entropy":
        #Fallback to cross entropy
        loss_function = nn.CrossEntropyLoss()
    # elif ...
    return loss_function