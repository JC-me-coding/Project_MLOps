import timm

NUM_CLASSES = 5

def make_model(model_name, pretrained=True):
    net = timm.create_model(model_name, num_classes=NUM_CLASSES, pretrained=pretrained)
    return net