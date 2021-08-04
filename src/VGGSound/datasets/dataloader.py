import os
# import cv2
import json
import torch
import csv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from PIL import Image
import glob
import sys
from scipy import signal
import random
import soundfile as sf

class GetAudioVideoDataset(Dataset):

    def __init__(self, args, mode='train', transforms=None):
        data2path = {}
        classes = []
        classes_ = []
        data = []
        data2class = {}

        with open(args.csv_path + 'sound_stat.csv') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                classes.append(row[0])

        file_path = args.train if mode=='train' else args.test
        with open(args.csv_path  + file_path) as f:
            csv_reader = csv.reader(f)
            for item in csv_reader:
                #if item[1] in classes and os.path.exists(args.data_path + item[0][:-3] + 'wav'):
                filename = args.data_path + item[0] + '_' + str(int(float(item[1]))*1000) + '_' + str(int(float(item[2]))*1000) + '.flac'
                if item[3] in classes and os.path.exists(filename):
                    #data.append(item[0])
                    data.append(filename)
                    #data2class[item[0]] = item[1]
                    data2class[filename] = item[3]

        self.audio_path = args.data_path 
        self.mode = mode
        self.transforms = transforms
        self.classes = sorted(classes)
        self.data2class = data2class

        # initialize audio transform
        self._init_atransform()
        #  Retrieve list of audio and video files
        self.video_files = []
        
        for item in data:
            self.video_files.append(item)
        print('# of audio files = %d ' % len(self.video_files))
        print('# of classes = %d' % len(self.classes))


    def _init_atransform(self):
        self.aid_transform = transforms.Compose([transforms.ToTensor()])


    def __len__(self):
        return len(self.video_files)  

    def __getitem__(self, idx):
        wav_file = self.video_files[idx]
        # Audio
        # samples, samplerate = sf.read(self.audio_path + wav_file[:-3]+'wav')
        with open(wav_file, 'rb') as f:
            samples, samplerate = sf.read(f, always_2d=True)

        # repeat in case audio is too short
        #print('shape 1', samples.shape)
        idx = np.random.randint(0, np.maximum(1, np.minimum(700000, samples.shape[0]-160000)))
        samples = samples[idx:idx+160000]
        #print('shape 2', samples.shape)
        resamples = np.tile(samples.T[0],10)[:160000]
        # print(resamples.shape, samples.shape)
        # exit()

        resamples[resamples > 1.] = 1.
        resamples[resamples < -1.] = -1.
        frequencies, times, spectrogram = signal.spectrogram(resamples, samplerate, nperseg=512, noverlap=353)
        spectrogram = np.log(spectrogram + 1e-7)

        mean = np.mean(spectrogram)
        std = np.std(spectrogram)
        spectrogram = np.divide(spectrogram-mean,std+1e-9)

        return spectrogram, resamples,self.classes.index(self.data2class[wav_file]), '_'.join(wav_file.split('/')[-1].split('_')[:-2])


