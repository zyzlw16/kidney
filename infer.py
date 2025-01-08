import logging
import os
import sys

import numpy as np
import torch
import csv
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from logger import get_logger

import monai
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.transforms import Activations, AsDiscrete, Compose, LoadImaged, RandRotate90d, Resized, ScaleIntensityRanged, \
    RandShiftIntensityd, RandFlipd, NormalizeIntensityd, Spacingd, RandScaleIntensityd
from monai.networks.nets import densenet
from monai.utils import set_determinism

#from swinMM import SSLHead, load_pretrained_model
from monai.losses import ContrastiveLoss

from monai.networks.nets import densenet, resnet, vitautoenc
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import pandas as pd
import json

def load_pre_trained(model, path):
   
    print('Loading pre-trained weights!')
    pretrained_dict = torch.load(path, map_location='cpu')

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    return model


def load_json(filename):
    with open(filename, 'r+', encoding='utf-8') as fp:
        data_dict = json.load(fp)
    return data_dict

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(42)
    json_path = '/media/lab/data4/kidney/code/json_files/'
    model_name ='radio_CBloss1'#'clinic'#'radio_patho_clinic'#'clinic'#'radio_patho_clinic'#'clinic' #'radio_save'#'patho_save'
    model_weight_path = f'/media/lab/data4/kidney/code/model_weight/{model_name}'
    if not os.path.exists(model_weight_path):
        os.makedirs(model_weight_path)
    logger = get_logger(model_weight_path + f'/result_{model_name}.log')
    df = pd.read_csv('/media/lab/data4/kidney/code/json_files/label.csv')
    for idx in range(3,4):
        train_map = {}
        valid_map = {}
        data_name = load_json(json_path + f'data_split_{4}.json')
        
        logger.info(f"data_split_{4}.json")
        
        with open(json_path + 'label.csv', 'r', newline='',  encoding='gbk') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row["path"].strip() in data_name['train']:

                    image = row["path"].strip()
                    label = row["label"].strip()
                    train_map[image] = {'label': label}

                elif row["path"].strip() in data_name['valid']:
                    image = row["path"].strip()
                   
                    label = row["label"].strip()

                    valid_map[image] = {'label': label}

        train_files = [{"img": key, "label": np.array(value['label'], dtype=int)} for key, value in train_map.items()]
        val_files = [{"img": key, "label": np.array(value['label'], dtype=int)} for key, value in valid_map.items()]

        print(len(train_files), len(val_files))

        val_transforms = Compose(
            [
                LoadImaged(keys=["img"], ensure_channel_first=True),
                ScaleIntensityRanged(
                keys=["img"],
                a_min=-100,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True,
                ),
                Spacingd(
                    keys=["img"],
                    pixdim=(0.78, 0.78, 0.78),
                    mode=("bilinear"),
                    allow_missing_keys=True
                ),
                Resized(keys=["img"], spatial_size=(384, 384, 128)),
            ]
        )
        post_pred = Compose([Activations(softmax=True)])
        post_label = Compose([AsDiscrete(to_onehot=2)])

        # create a validation data loader
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, pin_memory=torch.cuda.is_available())
        # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = densenet.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)

        pre_trained_path = model_weight_path + f"/best_metric_model_resnet_t2_dict_{idx}.pth"
        model = load_pre_trained(model=model, path = pre_trained_path)

        auc_metric = ROCAUCMetric()
        model.eval()
        with torch.no_grad():
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y = torch.tensor([], dtype=torch.long, device=device)
            y_label = []
            y_pred_act0 = []
            y_pred_act1 =[]
            for i , val_data in enumerate(val_loader):
                val_images = val_data["img"].to(device), 
                val_labels = torch.tensor([int(x) for x in val_data["label"]]).to(device)
                y_pred = torch.cat([y_pred,model(val_images[0])], dim=0)
                y = torch.cat([y, val_labels], dim=0)
                y_label.append(val_data["img"].meta["filename_or_obj"][0].split('/')[-1])
                
            acc_value = torch.eq(y_pred.argmax(dim=1), y)
            acc_metric = acc_value.sum().item() / len(acc_value)
            y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
            y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
            y_pred_act1 = [post_pred(i).cpu().numpy() for i in decollate_batch(y_pred)]
            auc_metric(y_pred_act, y_onehot)
            auc_result = auc_metric.aggregate()
            auc_metric.reset()
            print(
                    " current accuracy: {:.4f} current AUC: {:.4f} ".format(
                            acc_metric, auc_result)
                        )
            pred = np.array(y_pred_act1)
            label = y.detach().cpu().numpy()
            with open( f'/media/lab/data4/kidney/code/model_weight/result/val.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['ID', 'label', 'pre'])

                for row in zip(y_label, label, pred[:,1]):
                    writer.writerow(row)
if __name__ == "__main__":
    main()