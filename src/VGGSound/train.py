import os
import torch
from torch.optim import *
import torchvision
from torchvision.transforms import *
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import json
import argparse
import csv
from model import AVENet
from datasets import GetAudioVideoDataset




def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        default='/scratch/shared/beegfs/hchen/train_data/VGGSound_final/audio/',
        type=str,
        help='Directory path of data')
    parser.add_argument(
        '--save_path',
        default='/local1/emazuh/elk/vggsound/saved_models/',
        type=str,
        help='Directory path of results')
    parser.add_argument(
        '--summaries',
        default='/local1/emazuh/elk/vggsound/models/vggsound_netvlad.pth.tar',
        type=str,
        help='Directory path of pretrained model')
    parser.add_argument(
        '--pool',
        default="vlad",
        type=str,
        help= 'either vlad or avgpool')
    parser.add_argument(
        '--csv_path',
        default='./data/',
        type=str,
        help='metadata directory')
    parser.add_argument(
        '--train',
        default='train.csv',
        type=str,
        help='train csv files')
    parser.add_argument(
        '--batch_size', 
        default=32, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_classes',
        default=309,
        type=int,
        help=
        'Number of classes')
    parser.add_argument(
        '--new_n_classes',
        default=10,
        type=int,
        help=
        'Change number of classes in pretrained model by adding a new FC layer')
    parser.add_argument(
        '--model_depth',
        default=18,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    return parser.parse_args() 



def main():
    args = get_arguments()

    # create prediction directory if not exists
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    LATEST_PATH = args.save_path + "latest.pth"
    BEST_PATH = args.save_path + "best.pth"
    # init network
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model= AVENet(args) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    
    # load pretrained models
    checkpoint = torch.load(args.summaries)
    model.load_state_dict(checkpoint['model_state_dict'])
    # update model to detect elk sound or not
    model.audnet.fc_ = nn.Linear(model.audnet.penum, args.new_n_classes)
    model.to(device)
    print('load pretrained model.')

    # create dataloader
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset = GetAudioVideoDataset(args,  mode='train')
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, 
                                               sampler=train_sampler, num_workers = 12)
    validation_dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                                    sampler=valid_sampler, num_workers = 12)
    
#     softmax = nn.Softmax(dim=1)
#     print("Loaded dataloader.")

#     print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    softmax = nn.Softmax(dim=1)
    best_loss = 1e10
        
    for epoch in range(50):  # loop over the dataset multiple times

        model.train()
        running_loss = 0.0
        train_loss = 0.0
        for step, (spec, audio, label, name) in enumerate(train_dataloader):

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            spec = Variable(spec).cuda()
            label = Variable(label).cuda()
            outputs = softmax(model(spec.unsqueeze(1).float()))
        
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            # print statistics
            
            running_loss += loss.item()
#             if step % 60 == 59:    # print every 2000 mini-batches
#                 print('[%d, %5d] train loss: %.3f' %
#                       (epoch + 1, step + 1, running_loss / 60))
#                 running_loss = 0.0

        val_loss = running_loss / step
        print('[%d, %5d] train loss: %.3f' %
              (epoch + 1, step + 1, val_loss))        
                
        # validation
        model.eval()
        val_loss = 0.0
        running_loss = 0.0
        for step, (spec, audio, label, name) in enumerate(validation_dataloader):

            # forward + backward + optimize
            spec = Variable(spec).cuda()
            label = Variable(label).cuda()
            outputs = softmax(model(spec.unsqueeze(1).float()))
        
            loss = criterion(outputs, label)
            
            running_loss += loss.item()
            
        val_loss = running_loss / step
        print('[%d, %5d] val loss: %.3f' %
              (epoch + 1, step + 1, val_loss))
        running_loss = 0.0

        if val_loss < best_loss:
            torch.save(model.state_dict(), BEST_PATH)
        torch.save(model.state_dict(), LATEST_PATH)
        
#     model.eval()
#     for step, (spec, audio, label, name) in enumerate(testdataloader):
#         print('%d / %d' % (step,len(testdataloader) - 1))
#         spec = Variable(spec).cuda()
#         label = Variable(label).cuda()
#         aud_o = model(spec.unsqueeze(1).float())
# #         aud_o = model(spec.unsqueeze(1).squeeze(-1).float())

#         prediction = softmax(aud_o)

#         for i, item in enumerate(name):
#             np.save(args.result_path + '/%s.npy' % item,prediction[i].cpu().data.numpy())

if __name__ == "__main__":
    main()

