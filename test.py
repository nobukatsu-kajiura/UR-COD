import torch
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np
import os
import argparse
from model.URCOD import Generator
from utils.dataloader import test_dataset
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='UR-SINetv2-pretrained')
parser.add_argument('--dataset', type=str, default='SINetv2')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--latent_dim', type=int, default=3, help='latent dim')
parser.add_argument('--feat_channel', type=int, default=32, help='reduced channel of camouflaged feat')
opt = parser.parse_args()

torch.cuda.set_device(opt.gpu)
generator = Generator(channel=opt.feat_channel, latent_dim=opt.latent_dim)
generator.load_state_dict(torch.load('./checkpoints/{}/Model_{}_gen.pth'.format(opt.name, opt.epoch)))
generator.cuda()
generator.eval()

dataset_path = './data/TestDataset/'
dataset_list = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']

for dataset_name in dataset_list:
    save_path = './results/{}/{}epoch/{}/'.format(opt.name, opt.epoch, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    image_root = dataset_path + '{}/Img_RGB/'.format(dataset_name)
    pseudomask_root = dataset_path + '{}/Pseudo_Mask/{}/'.format(dataset_name, opt.dataset)

    test_loader = test_dataset(image_root, pseudomask_root, opt.testsize)
    for i in tqdm(range(test_loader.size)):
        image, pseudomask, HH, WW, name = test_loader.load_data()
        image = image.cuda()
        pseudomask = pseudomask.cuda()
        generator_pred, pseudoedge = generator.forward(image, pseudomask, training=False)
        generator_pred = F.upsample(generator_pred, size=[WW,HH], mode='bilinear', align_corners=False)
        generator_pred = generator_pred.sigmoid().data.cpu().numpy().squeeze()
        cv2.imwrite(save_path + name, generator_pred*255)
