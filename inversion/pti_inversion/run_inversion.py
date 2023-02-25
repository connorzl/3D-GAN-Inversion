import sys
sys.path.append("../3dgan")

from torch.utils.tensorboard import SummaryWriter
from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config, hyperparameters
import shutil

from pti_training.coaches.latent_id_coach import LatentIDCoach
from utils.ImagesDataset import ImagesDataset, GrayscaleImagesDataset, DECADataset
from utils.parse_args import parse_args

def run_PTI(run_name='', use_wandb=False):
    parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    if hyperparameters.temporal_consistency_loss:
        assert hyperparameters.batch_size % 2 == 0, 'Batch size must be divisible by two for temporal consistency'

    dataset = DECADataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
        pose_path=paths_config.input_pose_path)

    dataloader = DataLoader(dataset, batch_size=hyperparameters.batch_size, shuffle=False)
    coach = LatentIDCoach(dataloader, use_wandb)
    coach.train()
    return global_config.run_name


if __name__ == '__main__':
    run_PTI(run_name='')
