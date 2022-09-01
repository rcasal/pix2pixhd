import argparse
import os
import warnings
from sampling.sampling import sample_images
from utils.utils import str2bool
from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def parse_args():
    desc = "Pix2PixHD_sample"

    parser = argparse.ArgumentParser(description=desc)

    # Dataset parameters and input paths
    parser.add_argument('--n_classes', type=int, default=2, help='Number of segmented instances in the dataset. Eg. Character and background')
    parser.add_argument('--n_features', type=int, default=3, help='Number of channels. Eg. 3 for RGB')
    parser.add_argument('--input_path_dir', type=str, default="/Users/ramirocasal/Documents/Datasets/sword_sorcery_data_for_ramiro/test_dataset", help='Path root where inputs are located. By default it will contain 3 subfolders: img, inst, label')
    parser.add_argument('--input_img_dir', type=str, default="02_output", help='Folder name for input images located in input_path_dir')
    parser.add_argument('--input_label_dir', type=str, default="--nodir--", help='Folder name for optional labeled images located in input_path_dir')        # 01_segmented_input
    parser.add_argument('--input_inst_dir', type=str, default="01_segmented_input", help='Folder name for optional instances images located in input_path_dir') 
    parser.add_argument('--saved_model_path', type=str, default="Saved_Models", help='Path to the saved model')

    # Experiment parameters
    parser.add_argument('--experiment_name', type=str, default="", help='A name of the training experiment')
    parser.add_argument('--resume_training', type=str2bool, nargs='?', const=True, default=False, help="Continue training allows to resume training. You'll need to add experiment name args to identify the experiment to recover.")

    # Output parameters
    parser.add_argument('--target_width', type=int, default=1024, help='The size of the output image.')

    # Output paths
    parser.add_argument('--output_path_dir', type=str, default="", help='The base directory to hold the results')
    parser.add_argument('--output_images_path', type=str, default="Sampled_images", help='Folder name for save images during training')

    # Warnings parameters
    parser.add_argument('--warnings', type=str2bool, nargs='?', const=False, default=True, help="Show warnings")

    return parser.parse_args()


def main():
    args = parse_args()

    # warnings
    if args.warnings:
        warnings.filterwarnings("ignore")

    # Resume training and experiment name
    args.experiment_name = args.experiment_name
    
    # Output path dir
    args.output_path_dir = os.path.join(args.output_path_dir,args.experiment_name) 
    print('creating directories in ' + args.output_path_dir)
    os.makedirs(args.output_path_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_path_dir, args.output_images_path), exist_ok=True)

    args.saved_model_path = os.path.join(args.output_path_dir, args.saved_model_path, 'pix2pixHD_model.pth')

    sample_images(args)


if __name__ == '__main__':
    main()
