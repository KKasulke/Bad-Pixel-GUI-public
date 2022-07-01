import argparse
from re import S
from turtle import st
import numpy as np
import os
import torch
from tensorboardX import SummaryWriter
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm

import opt
from loss import InpaintingLoss
from net import PConvUNet
from net import VGG16FeatureExtractor
from places2 import Places2
from util.io import load_ckpt
from util.io import save_ckpt
import config as cfg


class InfiniteSampler(data.sampler.Sampler):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed()
                order = np.random.permutation(self.num_samples)
                i = 0

# get AI Model file name
def getLatestAIModelName(model_path):
    # check for valid directory name:
    if model_path == '' or not os.path.exists(model_path):
        return ''

    # check for files in directory:
    name = ''
    numFiles = len(os.listdir(model_path))
    for i in range(numFiles-1, -1, -1):
        name = os.listdir(model_path)[i]
        if name.endswith('.pth'):
            return name
    return ''

# get pixel dimensions from AI-Model-Filename
def getLatestAIModelIterationNr(AIModelName):
    if(AIModelName == ''):
        return ''
    part1 = AIModelName.split("-")
    if len(part1) > 1:
        return part1[0]
    return ''

def train(images_path, mask_path, model_path, newiterations, batch_size, image_resolution):

    parser = argparse.ArgumentParser()
    # training options
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_finetune', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--n_threads', type=int, default=2)
    parser.add_argument('--save_model_interval', type=int, default=100)
    parser.add_argument('--image_size', type=int, default=image_resolution)
    parser.add_argument('--resume', type=str, default=True)
    parser.add_argument('--finetune', action='store_true')
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    device = torch.device('cuda')

    size = (args.image_size, args.image_size)
    img_tf = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor(),
        transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    mask_tf = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor()])

    dataset_train = Places2(images_path, mask_path, img_tf, mask_tf, 'train')

    iterator_train = iter(data.DataLoader(
        dataset_train, batch_size=args.batch_size,
        sampler=InfiniteSampler(len(dataset_train)),
        num_workers=args.n_threads))
    print(len(dataset_train))
    try:
        model = PConvUNet().to(device)

        if args.finetune:
            lr = args.lr_finetune
            model.freeze_enc_bn = True
        else:
            lr = args.lr

        start_iter = 0
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

        if args.resume:
            # get last checkpoint
            if len(os.listdir(model_path)) > 0:
                checkpoint = os.path.join(model_path, os.listdir(model_path)[-1])
                start_iter = load_ckpt(
                    checkpoint, [('model', model)], [('optimizer', optimizer)])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Starting from iter ', start_iter)

        end_iteration = start_iter + newiterations
        for i in tqdm(range(start_iter, end_iteration)):
            if not cfg.trainThreadActive:
                return
            model.train()

            image, mask, gt = [x.to(device) for x in next(iterator_train)]
            output, _ = model(image, mask)
            loss_dict = criterion(image, mask, output, gt)

            loss = 0.0
            for key, coef in opt.LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                loss += value

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % args.save_model_interval == 0 or (i + 1) == end_iteration:
                # previous version: save_ckpt('{:s}/ckpt/{:d}.pth'.format(args.save_dir, i + 1),
                save_ckpt(os.path.join(model_path, f'{i+1:08}' + '-' + str(image_resolution) + '.pth'),
                        [('model', model)], [('optimizer', optimizer)], i + 1)

                # remove previous model versions to save disk space
                filenames = os.listdir(model_path)
                fileList = [os.path.join(model_path, filename) for filename in filenames]
                for removeindex in range(len(fileList) - 1):
                    os.remove(fileList[removeindex])

            # update progressBar
            cfg.trainProgress = 100*(i-start_iter+1)/(end_iteration-start_iter)
        cfg.trainFinishedSuccessful = 1
    except RuntimeError:
        cfg.trainFinishedSuccessful = -1
        return