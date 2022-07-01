import argparse
import torch
from torchvision import transforms

import opt
from places2 import Places2
from evaluation import evaluate
from net import PConvUNet
from util.io import load_ckpt
import os;

def test(images_paths, BPM_path, model_path, image_resolution, save_paths):
    parser = argparse.ArgumentParser()

    # training options
    parser.add_argument('--root', type=str, default=images_paths)
    parser.add_argument('--mask_root', type=str, default=BPM_path)
    parser.add_argument('--snapshot', type=str, default=model_path)
    parser.add_argument('--image_size', type=int, default=image_resolution)
    args = parser.parse_args()

    device = torch.device('cpu')

    size = (args.image_size, args.image_size)
    img_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor(),
        transforms.Normalize(mean=opt.MEAN, std=opt.STD)])
    mask_transform = transforms.Compose(
        [transforms.Resize(size=size), transforms.ToTensor()])

    # in github-code: parameter mask path is missing!
    dataset_val = Places2(args.root, args.mask_root, img_transform, mask_transform, 'test')

    model = PConvUNet().to(device)
    # get last checkpoint
    checkpoint = os.path.join(model_path, os.listdir(model_path)[-1])
    load_ckpt(checkpoint, [('model', model)])

    model.eval()
    evaluate(model, dataset_val, device, save_paths)
