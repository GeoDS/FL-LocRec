import argparse

import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from datasets import POIDataset, PrivatePOIDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd
import numpy as np

import copy
from functools import reduce

torch.set_default_tensor_type(torch.cuda.FloatTensor)

import syft as sy
hook = sy.TorchHook(torch)

# central crypto server
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

from dfm import DeepFactorizationMachineModel, DeepFactorizationMachineModel_sy

def add_model(dst_model, src_model):
    params1 = dst_model.state_dict().copy()
    params2 = src_model.state_dict().copy()
    with torch.no_grad():
        for name1 in params1:
            if name1 in params2:
                params1[name1] = params1[name1] + params2[name1]
                params1[name1] = params1[name1]
                # secret sharing + fix precision (commented for better performance in single-machine-based simulation)
#                 params1[name1] = params1[name1].fix_prec(precision_fractional=8).share(*user_worker_list, crypto_provider=crypto_provider).get()
#                 params1[name1] += params2[name1].fix_prec(precision_fractional=8).share(*user_worker_list, crypto_provider=crypto_provider).get()
#                 params1[name1] = params1[name1].float_prec().to(device)
    model = copy.deepcopy(dst_model)
    model.load_state_dict(params1, strict=False)
    return model


def scale_model(model, scale):
    params = model.state_dict().copy()
    scale = torch.tensor(scale)
    with torch.no_grad():
        for name in params:
            params[name] = params[name].type_as(scale) * scale
    scaled_model = copy.deepcopy(model)
    scaled_model.load_state_dict(params, strict=False)
    return scaled_model


def update(data, target, model, optimizer):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target.float())
    loss.backward()
    optimizer.step()
    return model


def train(model_list, optimizer_list, user_worker_list, remote_dataset, criterion, epoch, device):
    # update remote models
    nks = []
    for remote_index in range(len(user_worker_list)):
        model_list[remote_index].send(remote_dataset[remote_index][0][0].location)
        nk = 0
        for idx in range(len(remote_dataset[remote_index])):
            data, target = remote_dataset[remote_index][idx]
            model_list[remote_index] = update(data, target, model_list[remote_index], optimizer_list[remote_index])
            nk += data.shape[0]
        nks.append(nk)
    
    avg_model = None
    
    with torch.no_grad():
        idx = 0
        scaled_models = []
        for model in model_list:
            nk = nks[idx]
            scaled_models.append(scale_model(model.get(), nk / float(sum(nks))))
            idx += 1
        nr_models = len(model_list)
        if nr_models > 1:
            avg_model = reduce(add_model, scaled_models).to(device)
        else:
            avg_model = copy.deepcopy(scaled_models[0])
        for model in model_list:
            model.load_state_dict(avg_model.state_dict())
            
            
def test(evaluation_model, criterion, data_loader, device):
    targets, predicts = list(), list()
    total_loss = 0
    with torch.no_grad():
        count = 0
        for fields, target in tqdm(data_loader, smoothing=0, mininterval=1.0):
            fields, target = fields.to(device), target.to(device)
            y = evaluation_model(fields)
            total_loss += criterion(y, target.float()).item()
            count += 1
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    
    total_loss /= count
    
    return total_loss, roc_auc_score(targets, predicts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="./encoded_data/hex_7/train_encoded.csv")
    parser.add_argument("--val_path", type=str, default="./encoded_data/hex_7/val_encoded.csv")
    parser.add_argument("--test_path", type=str, default="./encoded_data/hex_7/test_encoded.csv")
    parser.add_argument("--save_path", type=str, default="./weights")
    parser.add_argument("--manual_seed", type=int, default=2021)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--user_batch_size", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=16)
    parser.add_argument("--mlp_dim", type=int, default=16)
    parser.add_argument("--weight_decay", type=int, default=1e-6)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--precision_fractional", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    # load data from .csv files
    train_dataset = pd.read_csv(args.train_path)
    valid_dataset = pd.read_csv(args.val_path)
    test_dataset = pd.read_csv(args.test_path)
    
    # data type casting
    train_dataset['friend_num'] = train_dataset['friend_num'].astype(int)
    train_dataset['follow_num'] = train_dataset['follow_num'].astype(int)
    train_dataset['interaction'] = train_dataset['interaction'].astype(int)
    valid_dataset['friend_num'] = valid_dataset['friend_num'].astype(int)
    valid_dataset['follow_num'] = valid_dataset['follow_num'].astype(int)
    valid_dataset['interaction'] = valid_dataset['interaction'].astype(int)
    test_dataset['friend_num'] = test_dataset['friend_num'].astype(int)
    test_dataset['follow_num'] = test_dataset['follow_num'].astype(int)
    test_dataset['interaction'] = test_dataset['interaction'].astype(int)
    
    # data concatenation
    all_dataset = pd.concat([train_dataset, valid_dataset, test_dataset])
    
    # get field dims
    field_dims = torch.Tensor(np.array([all_dataset['gender'].unique().shape[0],
                            all_dataset['friend_num'].unique().shape[0],
                            all_dataset['follow_num'].unique().shape[0],
                            all_dataset['hex7'].unique().shape[0],
                            7, # week
                            24, # hour
                            all_dataset['venueCategory'].unique().shape[0],
                            all_dataset['bus'].unique().shape[0],
                            all_dataset['subway'].unique().shape[0],
                            all_dataset['parking'].unique().shape[0],
                            all_dataset['crime'].unique().shape[0],
                            all_dataset['interaction'].unique().shape[0],
                           ]))
    
    # for validation and testing, we use centralized datasets
    total_valid_dataset = POIDataset(valid_dataset, field_dims)
    total_test_dataset = POIDataset(test_dataset, field_dims)
    valid_data_loader = DataLoader(total_valid_dataset, batch_size=args.batch_size, num_workers=8)
    test_data_loader = DataLoader(total_test_dataset, batch_size=args.batch_size, num_workers=8)
    
    # list of virtual user clients
    user_worker_list = []
    # list of assigned datasets for each virtual user client
    remote_dataset = []
    
    print('initializing virtual user clients and decentralized training datasets...')
    for user_id, user_df in tqdm(train_dataset.groupby(['userId'])):
        user_worker = sy.VirtualWorker(hook, id=f"user{user_id}")
        user_dataset = PrivatePOIDataset(user_df, user_id, field_dims)
        train_data_loader = DataLoader(user_dataset, batch_size=args.user_batch_size, shuffle=True)
        subset = []
        for batch_idx, (data,target) in enumerate(train_data_loader):
            data = data.to(args.device).send(user_worker)
            target = target.to(args.device).send(user_worker)
            subset.append((data, target))
        remote_dataset.append(subset)
        user_worker_list.append(user_worker)
    
    # models stored in virtual user clients
    models = []
    # optimizers for users' models
    optimizers = []
    print('initializing models and optimizers...')
    for i in tqdm(range(user_worker_list.__len__())):
        model = DeepFactorizationMachineModel_sy(field_dims.cpu().numpy().astype(np.int), embed_dim=args.embed_dim, mlp_dims=(args.mlp_dim, args.mlp_dim), dropout=args.dropout).to(args.device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        models.append(model)
        optimizers.append(optimizer)
    
    criterion = torch.nn.BCELoss()
    
    # model for evaluation
    evaluation_model = DeepFactorizationMachineModel(field_dims.cpu().numpy().astype(np.int), embed_dim=args.embed_dim, mlp_dims=(args.mlp_dim, args.mlp_dim), dropout=args.dropout).to(args.device)
    
    for epoch_i in range(args.num_epochs):
        print(f"[training] Epoch {epoch_i}")
        # training
        train(models, optimizers, user_worker_list, remote_dataset, criterion, epoch_i, args.device)
        
        # validation
        evaluation_model.load_state_dict(models[0].state_dict().copy())
        evaluation_model.eval()
        loss, auc = test(evaluation_model, criterion, valid_data_loader, args.device)
        print(f'[validation] Epoch: {epoch_i} validation auc: {auc}')
        print(f'[validation] Epoch: {epoch_i} validation loss: {loss}')
        if (epoch_i) % 10 == 0:
            loss, auc = test(evaluation_model, criterion, test_data_loader, args.device)
            torch.save(model, f'{args.save_path}/dfm_{epoch_i}.pth')
            print(f'[test] Epoch: {epoch_i} test auc: {auc}')
            print(f'[test] Epoch: {epoch_i} test loss: {loss}')
