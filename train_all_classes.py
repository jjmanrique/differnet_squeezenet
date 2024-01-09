import config as c
from train import train
from utils import load_datasets, make_dataloaders

train_set, test_set = load_datasets(c.dataset_path, c.class_name)
train_loader, test_loader = make_dataloaders(train_set, test_set)
model = train(train_loader, test_loader)
