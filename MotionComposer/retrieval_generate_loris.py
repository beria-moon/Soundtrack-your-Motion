import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import OrderedDict
import yaml
import time
import argparse
from pathlib import Path
import librosa
import soundfile as sf
# import noisereduce as nr
import sys
from d2m.retrieval_dataset import S25Dataset
from d2m.utils import save_sample, load_yaml_config
from d2m.loris_modules_retrieval import LORIS

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default='./configs/')
    parser.add_argument('--dataset', default='')
    args = parser.parse_args()
    return args


def generate(config):

    model_save_path = config['model_save_path']
    print('model_save_path:',model_save_path)
    sample_save_path = config['sample_save_path'] + config['ckpt_name'].split('.')[0].split('_')[1] + '_' + str(config['model']['embedding_scale']) + '_' + str(config['model']['diffusion_step'])
    print("sample_save_path:",sample_save_path)
    loris = LORIS(config['model']).cuda()

    loris_ckpt_path = os.path.join(model_save_path, config['ckpt_name'])
    print('loris_ckpt_path:',loris_ckpt_path)
  
    state_dict = torch.load(loris_ckpt_path, map_location="cpu")
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        # name = k[7:]
        # if "ffusion" in name:
            
        #     name = "cond_di"+name
        # if "r.lstm" in name:
        #     name = "vencode"+name
        # if "n.weight" in name:
        #     name = "cond_lin.weight"
        # if "n.bias" in name:
        #     name = "cond_lin.bias"

        # if "r.genre" in name:
        #     name = "gencode"+name
        
  
        new_state_dict[name] = v

    loris.load_state_dict(new_state_dict, strict=True)
    print("*******Finish model loading******")

    #### create data loader ####
    test_dataset = S25Dataset(audio_files=config['audio_test_path'],retrieval_audio_files=config['audio_test_retrieval_path'], video_files=config['video_test_path'], motion_files=config['motion_test_path'], genre_label=config['genre_test_path'], augment=False, config=config['model'])
    va_loader = DataLoader(test_dataset, batch_size=2)
    print("*******Finish data loader*******")

    #### generate samples ####
    torch.backends.cudnn.benchmark = True
    for j, input in enumerate(va_loader):
        input['music'],input['refer_music'], input['motion'], input['video'] = input['music'].cuda(), input['refer_music'].cuda(),input['motion'].cuda(), input['video'].cuda()
        filename = input['filename']
        print(filename)
      
        sys.stdout.flush()
        audio_gen_batch = loris.sample(input)
        for i in range(audio_gen_batch.size(0)):
            sys.stdout.flush()
            audio_gen = audio_gen_batch[i]
            print('debug audio_gen:',audio_gen.shape)
  
            audio_gt = input['music'][i].squeeze().cpu()
            # print('debug audio_gt',audio_gt.shape)
            print(audio_gt.shape[0],audio_gen.shape[1])
            comp_len = audio_gt.shape[0] - audio_gen.shape[1]
            audio_gen = torch.cat((audio_gen, audio_gen[:, audio_gen.shape[1]-comp_len:]), -1)
            print('debug audio_gen:',audio_gen.shape)
          
            print("Generating testing sample:",filename[i],flush=True)
            #print("Generating testing sample:", j*audio_gen_batch.size(0)+i+1, flush=True)
            if not os.path.exists(sample_save_path):
                os.makedirs(sample_save_path)
            sample_gt = 'gt_' + filename[i] + '.wav'
            #sample_gt = 'gt_' + str(j*audio_gen_batch.size(0)+i+1) + '.wav'
            sample_gt = os.path.join(sample_save_path, sample_gt)
            print('debug',sample_gt)
            exit()
            sf.write(sample_gt, audio_gt.detach().cpu().numpy(), 22050)
            audio_gen = audio_gen.transpose(0, 1).squeeze().detach().cpu().numpy()

            # audio_gen = nr.reduce_noise(y=audio_gen, sr=22050)
            sample_gen = 'generated_'+ filename[i] + '.wav'
            #sample_gen = 'generated_'+ str(j*audio_gen_batch.size(0)+i+1) + '.wav'
            sample_gen = os.path.join(sample_save_path, sample_gen)
            sf.write(sample_gen, audio_gen, samplerate=22050)
            

    print("*******Finish generating samples*******")


def main():
    args = parse_args()

    # config_dir = os.path.join(args.config_path, args.dataset+'.yaml')
    config_dir = os.path.join(args.config_path,"retrieval_fs_s25.yaml")
    config = load_yaml_config(config_dir)
    generate(config)

if __name__ == '__main__':
    main()

