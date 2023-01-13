from src.data.dataloader import load_data
from omegaconf import OmegaConf
config = OmegaConf.load('src/model_config.yaml')
root = "data/processed/landscapes"
batch_size = config.data.batch_size
input_size = config.data.input_size

def test_data_split_size():
  train_loader = load_data(root, "train", batch_size, config.data)
  valid_loader = load_data(root, "val", 1, config.data)
  assert len(train_loader.dataset) == 10000
  assert len(valid_loader.dataset) == 1500

def test_img_shape():
  train_loader = load_data(root, "train", batch_size, config.data)
  valid_loader = load_data(root, "val", 1, config.data)
  for i, data in enumerate(train_loader):
    images, labels = data
    assert list(labels.size()) == [batch_size]
    assert images.shape == (batch_size,3,input_size,input_size)
    break
  for i, data in enumerate(valid_loader):
    images, labels = data
    assert images.shape == (1,3,input_size,input_size)
    break

