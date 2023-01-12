from omegaconf import OmegaConf
from src.model import make_model
import torch

def predict_input(model_weights, image):
  config = OmegaConf.load('src/model_config.yaml')
  classes = config.data.classes
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  image = image.to(device)
  net = make_model('predict', pretrained=True).to(device)
  net.load_state_dict(torch.load(model_weights))
  net.eval()
  with torch.no_grad():
    prediction = net(image)
    sm = torch.max(prediction, 1)
    print(classes[sm[1]], sm[0])

# root = "data/processed/landscapes"
# assert os.path.exists(root), "Download and extract the data first, using the script in src/data/data.py"
# valid_loader = load_data(root, split="val", batch_size = 1)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# for step, data in enumerate(valid_loader):
#     images, labels = data
#     images, labels = images.to(device), labels.to(device)
#     net = make_model('resnet34', pretrained=True).to(device)
#     torch.save(net.state_dict(), f'./models/model_{epoch}.pth')
#     #net.load_state_dict(torch.load(model_weights))
#     net.eval()
#     with torch.no_grad():
#       prediction = net(images)
#       sm = torch.max(prediction, 1)
#       print(sm[0], valid_loader.dataset.classes[sm[1]])
#     break