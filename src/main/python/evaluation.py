import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import config as cfg
from util.image import unnormalize


def evaluate(model, dataset, device, filename):
    for i in range(len(dataset)):
        image, mask, gt = zip(*[dataset[i]])
        image = torch.stack(image)
        mask = torch.stack(mask)
        gt = torch.stack(gt)
        with torch.no_grad():
            output, _ = model(image.to(device), mask.to(device))
        output = output.to(torch.device('cpu'))
        output_comp = (mask) * image + (1-mask) * output

        grid = make_grid(
            torch.cat((unnormalize(image), mask, unnormalize(output),
                    unnormalize(output_comp), unnormalize(gt)), dim=0))
        save_image(unnormalize(output_comp), filename[i])
        cfg.Ladebalken = cfg.Ladebalken + 1
