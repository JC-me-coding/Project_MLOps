from src.data.dataloader import load_data
from omegaconf import OmegaConf
config = OmegaConf.load('src/model_config.yaml')
root = "data/processed/landscapes"
batch_size = config.data.batch_size

def test_data():
  train_loader = load_data(root, split="train", batch_size = batch_size)
  valid_loader = load_data(root, split="val", batch_size = 1)
  assert len(train_loader.dataset) == 10000
  assert len(valid_loader.dataset) == 1500
  for i, data in enumerate(train_loader):
    images, labels = data
    assert list(labels.size()) == [batch_size]
    assert images.shape == (32,3,224,224)
    break
  for i, data in enumerate(valid_loader):
    images, labels = data
    assert images.shape == (1,3,224,224)
    break

