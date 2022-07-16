import torch 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os


def print_device_name():
  cuda = torch.device('cuda:0')
  torch.cuda.set_device(0)
  return print('Exp is running in {} No.{}'.format(
      torch.cuda.get_device_name(torch.cuda.current_device()), torch.cuda.current_device()))


def save_tensor_images(image_tensor_fake, image_tensor_real, epoch, path):
    '''
    Function for visualizing images: Given a tensor of imagess, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    # fake
    image_tensor_fake = (image_tensor_fake + 1) / 2
    image_fake_unflat = image_tensor_fake.detach().cpu()
    image_fake_grid = make_grid(image_fake_unflat[:1], nrow=1)
    
    # real
    image_tensor_real = (image_tensor_real + 1) / 2
    image_real_unflat = image_tensor_real.detach().cpu()
    image_real_grid = make_grid(image_real_unflat[:1], nrow=1)   

    fig, axs = plt.subplots(2)
    axs[0].imshow(image_fake_grid.permute(1, 2, 0).squeeze())
    axs[1].imshow(image_real_grid.permute(1, 2, 0).squeeze())

    output_path = os.path.join(path,f"epoch_{epoch:04d}.jpg")
    fig.savefig(output_path)
    #plt.savefig()
    #plt.close()



