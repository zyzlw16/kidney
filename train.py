import pandas as pd

# df = pd.read_csv('json_files/Clinic.csv')
# patient_id = '12417792'
# patient_row = df[df['病人编号'] == patient_id]
# numbers = patient_row.iloc[0, 1:].values
# int_numbers = [int(num) for num in numbers]  # 列表推导式
# print(int_numbers)
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
import json
from balanced_loss import Loss

def load_json(filename):
    with open(filename, 'r+', encoding='utf-8') as fp:
        data_dict = json.load(fp)
    return data_dict

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    set_determinism(42)
    json_path = './json_files/'
    model_name ='radio_newww'#'clinic'#'radio_patho_clinic'#'clinic'#'radio_patho_clinic'#'clinic' #'radio_save'#'patho_save'
    model_weight_path = f'/media/lab/data4/kidney/code/model_weight/{model_name}'
    if not os.path.exists(model_weight_path):
        os.makedirs(model_weight_path)
    logger = get_logger(model_weight_path + f'/train_resnet_{model_name}1.log')
    df = pd.read_csv('/media/lab/data4/kidney/code/json_files/label.csv')
    for idx in range(0, 5):
        train_map = {}
        valid_map = {}
        data_name = load_json(json_path + f'data_split_{idx}.json')
        
        logger.info(f"data_split_{idx}.json")
        
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
        # 1. 找到 label 为 0 的样本
        zero_label_data = [item for item in train_files if item['label'] == 0]

        # 2. 复制这些数据。比如复制一倍，过采样倍数为2（可以根据需求调整）
        num_zeros = len(zero_label_data)
        oversample_ratio = 1  # 例如复制一倍
        oversampled_data = zero_label_data #* oversample_ratio

        # 3. 将复制的样本添加回原始数据
        train_files.extend(oversampled_data)
        val_files = [{"img": key, "label": np.array(value['label'], dtype=int)} for key, value in valid_map.items()]

        print(len(train_files), len(val_files))
        # Define transforms for image
        train_transforms = Compose(
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
                RandRotate90d(keys=["img"], prob=0.8, spatial_axes=[0,1]),
                RandShiftIntensityd(
                keys=["img"],
                offsets=0.10,
                prob=0.60,
            ),
                    #RandFlipd(keys=["img"], prob=0.5, spatial_axis=1, allow_missing_keys=True)
            ]
        )
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

        # Define dataset, data loader
        batch_size = 2
        check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        check_loader = DataLoader(check_ds, batch_size=batch_size, num_workers=1, pin_memory=torch.cuda.is_available())
        check_data = monai.utils.misc.first(check_loader)
        print(check_data["img"].shape, check_data["label"])

        # create a training data loader
        train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=torch.cuda.is_available())

        # create a validation data loader
        val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=torch.cuda.is_available())
        # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = densenet.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        #model = densenet.Densenet121(model_size="large").to(device)
        optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
       
        #loss_function = torch.nn.CrossEntropyLoss()
        loss_function = Loss(
            loss_type="focal_loss",
            samples_per_class=[60,419],
            class_balanced=True
        )
        auc_metric = ROCAUCMetric()

        # start a typical PyTorch training
        val_interval = 2
        best_metric = -1
        best_metric_acc = -1
        best_metric_epoch = -1
        writer = SummaryWriter()
        logger.info("start trainging....")
        for epoch in range(50):
            logger.info("-" * 10)
            logger.info(f"epoch {epoch + 1}/{50}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs= batch_data["img"].to(device)
                labels = torch.tensor([int(x) for x in batch_data["label"]]).to(device)
                label_1_indices = (labels == 1).nonzero(as_tuple=True)[0]

                p = np.random.uniform(0, 1)
                if p > 0.9 and len(label_1_indices) ==2:
                    continue
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size
                #print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
            epoch_loss /= step
            logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    y_pred = torch.tensor([], dtype=torch.float32, device=device)
                    y = torch.tensor([], dtype=torch.long, device=device)
                    for val_data in val_loader:
                        val_images = val_data["img"].to(device), 
                        val_labels = torch.tensor([int(x) for x in val_data["label"]]).to(device)
                    
                        y_pred = torch.cat([y_pred, model(val_images[0])], dim=0)
                        y = torch.cat([y, val_labels], dim=0)

                    acc_value = torch.eq(y_pred.argmax(dim=1), y)
                    acc_metric = acc_value.sum().item() / len(acc_value)
                    y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                    y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                    auc_metric(y_pred_act, y_onehot)
                    auc_result = auc_metric.aggregate()
                    auc_metric.reset()
                    del y_pred_act, y_onehot
                    if auc_result > best_metric:
                        best_metric = auc_result
                        best_metric_acc = acc_metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), model_weight_path + f"/best_metric_model_resnet_t2_dict_{idx}.pth")
                        logger.info("saved new best metric model")
                    logger.info(
                        "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best aucuracy: {:.4f} at epoch {}".format(
                            epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                        )
                    )
                    writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
        logger.info(f"train completed, best_metric: AUC-{best_metric:.4f} ACC-{best_metric_acc:.4f}at epoch: {best_metric_epoch}")
        #torch.save(model.state_dict(), weight_path + f"laster_metric_model_resnet_t2_dict_{idx}.pth")
        writer.close()
        #             if auc_result > best_metric:
        #                 best_metric = auc_result
        #                 best_metric_acc = acc_metric
        #                 best_metric_epoch = epoch + 1
        #                 torch.save(model.state_dict(), model_weight_path + f"/best_metric_model_resnet_multi_dict_{idx}.pth")
        #                 logger.info("saved new best metric model")
        #             logger.info(
        #                 "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best aucuracy: {:.4f} at epoch {}".format(
        #                     epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
        #                 )
        #             )
        #             writer.add_scalar("val_accuracy", acc_metric, epoch + 1)
        # logger.info(f"train completed, best_metric: AUC-{best_metric:.4f} ACC-{best_metric_acc:.4f}at epoch: {best_metric_epoch}")
        # #torch.save(model.state_dict(), model_weight_path + f"/best_metric_model_resnet_multi_dict_{idx}.pth")
        # writer.close()


if __name__ == "__main__":
    main()
