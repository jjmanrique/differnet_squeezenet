import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import config as c
from test import test
from utils import *

auc = list()
times = list()
classes = list()
file_names = list()

for f in os.scandir('./models/'):
    fname = f.name
    if '_fc' in fname:
        if not 'metal_nut' in fname:
            class_name = fname.split('_')[1]
        else:
            class_name = 'metal_nut'
        print(class_name)
        train_set, test_set = load_datasets(c.dataset_path, class_name)
        _, test_loader = make_dataloaders(train_set, test_set)
        
        for i in tqdm(range(3)):
            temp_roc = list()
            temp_time = list()
            roc, pred_time = test(test_loader,fname, output = True)
            temp_roc.append(roc)
            temp_time.append(pred_time)
        times.append(np.mean(temp_time))
        auc.append(max(temp_roc))
        classes.append(class_name)
        file_names.append(fname)

out = pd.DataFrame({'class': classes, 'file': file_names,'auc': auc, 'time': times})

out.to_csv("./results.csv", index = False)  