import argparse

import os
import torch
import wandb

import numpy as np
from itertools import chain

from tqdm import tqdm, trange
import torchvision.transforms as T

from models.ad_models import FeatureExtractors
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big
from models.datset_2D import LowLightDataset, AdditiveGaussianNoiseTransform, GlobalLowLightTransform, KorniaLowLightWithShadowTransform, PatchLowLightTransform, get_dataloader


def set_seeds(sid=115):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def train(args):
    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project = 'AD',
        name = model_name
    )
    common = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    lowlight = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        # ColorShiftLowLightTransform(),
        # ContrastReductionLowLightTransform(),
        # PatchLowLightTransform(),
        KorniaLowLightWithShadowTransform(p=1.0),
        # RandomLowLightTransform(),
        # VignetteLowLightTransform(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # Dataloader.
    train_loader = get_dataloader("./", )
    
    # Feature extractors.
    feature_extractor = FeatureExtractors()

    # Model instantiation. 
    FAD_LLToClean = FeatureProjectionMLP(in_features = 768, out_features = 768)
    

    optimizer = torch.optim.Adam(params = chain(FAD_LLToClean.parameters()))

    FAD_LLToClean.to(device)

    metric = torch.nn.CosineSimilarity(dim = -1, eps = 1e-06)

    for epoch in trange(args.epochs_no, desc = f'Training Feature Transfer Net.{args.class_name}'):
        FAD_LLToClean.train()
        epoch_cos_sim = []
        for i, (images, lowlight) in enumerate(tqdm(train_loader, desc = f'Epoch {epoch + 1}/{args.epochs_no}')):
            images, lowlight = images.to(device), lowlight.to(device)

            features,features_lowlight = feature_extractor(images, lowlight)

            transfer_features = FAD_LLToClean(features_lowlight)

            loss = 1- metric(features, transfer_features).mean()

            epoch_cos_sim.append(loss.item())
            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        wandb.log({
            "Epoch": epoch + 1,
            "Loss": np.mean(epoch_cos_sim),
        })
        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)
        if (epoch + 1) % args.save_interval == 0:
            torch.save(FAD_LLToClean.state_dict(), rf'{args.checkpoint_folder}/{args.class_name}/FAD_LLToClean_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train Crossmodal Feature Networks (FADs) on a dataset.')

    parser.add_argument('--dataset_path', default = './datasets/', type = str, 
                        help = 'Dataset path.')

    parser.add_argument('--checkpoint_savepath', default = './checkpoints/checkpoints_FAD_mvtec', type = str, 
                        help = 'Where to save the model checkpoints.')
    
    # parser.add_argument('--class_name', default = None, type = str, choices = ["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope", "tire"],
    #                     help = 'Category name.')
    
    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train the FADs.')

    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')
    
    parser.add_argument('--save_interval', default = 5, type = int,
                        help = 'Number of epochs to train the FADs.')

    args = parser.parse_args()
    train(args)