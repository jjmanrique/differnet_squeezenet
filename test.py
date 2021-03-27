from sys import argv
from time import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import config as c
from model import DifferNet, load_weights
from utils import *

def test(test_loader,model_path):
    model = DifferNet()
    model = load_weights(model, model_path)
    model.to(c.device)
    model.eval()

    test_loss = list()
    test_z = list()
    test_labels = list()
    times = list()
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader, disable=c.hide_tqdm_bar)):
            inputs, labels = preprocess_batch(data)
            t0 = time()
            z = model(inputs)
            t1 = time()
            loss = get_loss(z, model.nf.jacobian(run_forward=False))
            test_z.append(z)
            test_loss.append(t2np(loss))
            test_labels.append(t2np(labels))
            times.append(t1-t0)

    test_loss = np.mean(np.array(test_loss))
    if c.verbose:
        print('test_loss: {:.4f}'.format(test_loss))

    test_labels = np.concatenate(test_labels)
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

    z_grouped = torch.cat(test_z, dim=0).view(-1, c.n_transforms_test, c.n_feat)
    anomaly_score = t2np(torch.mean(z_grouped ** 2, dim=(-2, -1)))
    roc = roc_auc_score(is_anomaly, anomaly_score)
    print('AUROC is : ',round(roc,2))
    print('mean predict time is ', round(np.mean(times),2))



if __name__ == '__main__':
    train_set, test_set = load_datasets(c.dataset_path, c.class_name)
    _, test_loader = make_dataloaders(train_set, test_set)
    test(test_loader,argv[1])