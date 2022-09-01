import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from tqdm.auto import tqdm
import time
import os

from models.discriminators import MultiscaleDiscriminator
from models.generators import LocalEnhancer
from models.loss import gd_loss, VGG_Loss
from models.models_utils import Encoder
from utils.dataloader import DataLoader, SwordSorceryDataset
from utils.utils import save_tensor_images
from torch.utils.tensorboard import SummaryWriter


NODES      = int(os.environ.get('WORLD_SIZE', 1))

# Parse torch version for autocast
# ######################################################
version = torch.__version__
version = tuple(int(n) for n in version.split('.')[:-1])
has_autocast = version >= (1, 6)
# ######################################################

def sample_images(args):
    
    # init params
    n_classes = args.n_classes                  # total number of object classes # maybe we're not using n_classes, just args.n_classes
    rgb_channels = n_features = args.n_features       
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    # Training directories
    dataset_dir = {
        'path_root': args.input_path_dir, 
        'path_inputs': {
            'input_img': args.input_img_dir,
            'inst_map': args.input_inst_dir,
            'label_map': args.input_label_dir, 
            }
        }

    dataloader = DataLoader(
        SwordSorceryDataset(dataset_dir, target_width=args.target_width, n_classes=n_classes),
        collate_fn=SwordSorceryDataset.collate_fn, batch_size=1, shuffle=False, drop_last=False, pin_memory=True,
    )
    encoder = Encoder(rgb_channels, n_features).to(args.device)
    generator = LocalEnhancer(dataloader.dataset.get_input_size_g(), rgb_channels)
    discriminator = MultiscaleDiscriminator(dataloader.dataset.get_input_size_d())

    # Model and output paths
    args.output_images_path = os.path.join(args.output_path_dir, args.output_images_path)

    # Sampling
    generator, discriminator = generator.to(args.gpu).eval(), discriminator.to(args.gpu).eval()
   
    # recover model
    cp = torch.load(args.saved_model_path)
    #encoder.load_state_dict(cp['encoder_state_dict'])          # Load state of the last epoch
    generator.load_state_dict(cp['generator_state_dict'])
    discriminator.load_state_dict(cp['discriminator_state_dict'])

    for (img_i, labels, insts, bounds, _) in dataloader:
        img_i = img_i.cuda(args.gpu)
        labels = labels.cuda(args.gpu)
        insts = insts.cuda(args.gpu)
        bounds = bounds.cuda(args.gpu)
        
        img_o_fake = generator(torch.cat((img_i,labels, bounds), dim=1))

        save_tensor_images(img_o_fake.to(img_i.dtype), args.output_images_path)

    return print("done training")


# Freeze encoder and wrap to support high-resolution inputs/outputs
def freeze(encoder):
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    @torch.jit.script
    def forward(x, inst):
        x = F.interpolate(x, scale_factor=0.5, recompute_scale_factor=True)
        inst = F.interpolate(inst.float(), scale_factor=0.5, recompute_scale_factor=True)
        feat = encoder(x, inst.int())
        return F.interpolate(feat, scale_factor=2.0, recompute_scale_factor=True)
    return forward
