import argparse
from email.policy import default

from torch.profiler import profile, record_function, ProfilerActivity
import os
import torch
import wandb

from HWMNet.WT.transform import DWT, IWT
import numpy as np
from itertools import chain

from tqdm import tqdm, trange
import torchvision.transforms as T

from models.ad_models import FeatureExtractors
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big
from dataset2D import *
from models.features2d import Multimodal2DFeatures
# from models.dataset import BaseAnomalyDetectionDataset
from HWMNet.model.HWMNet import HWMNet
from models.node_module import NodeModule, ChannelAttentionReduction

def set_seeds(sid=42):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def train(args):
    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"FAD_LLToClean{args.person}{args.unique_id}.pth"


    wandb.init(project="AD", name=model_name)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    common = T.Compose(
        [
            SquarePad(),
            T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # Dataloader.
    train_loader = get_dataloader(
        os.path.join(args.dataset_path, args.class_name, "normal"), common, common, 4, 16, True)

    # Feature extractors.
    feature_extractor = Multimodal2DFeatures()
    # feature_level_light = HWMNet()
    feature_level_light=HWMNet(in_chn=3, wf=64, depth=2)
    node_module = NodeModule()
    sa = ChannelAttentionReduction()
    # Model instantiation.
    FAD_LLToClean = FeatureProjectionMLP(in_channels=192, out_channels=192)

    # optimizer = torch.optim.Adam(params=chain(FAD_LLToClean.parameters()))

    optimizer = torch.optim.Adam(
        list(node_module.parameters()) + list(feature_level_light.parameters()) + list(sa.parameters()),  # Chain all parameters into a single list
        lr=1e-4,  # Learning rate
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,  # Optional L2 regularization
    )


    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma=0.96)
    # FAD_LLToClean.to(device)
    feature_extractor.to(device)

    feature_level_light.to(device)
    node_module.to(device)
    sa.to(device)


    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    for epoch in trange(
        args.epochs_no, desc=f"Training Feature Transfer Net.{args.class_name}"
    ):
        # FAD_LLToClean.train()
        feature_level_light.train()
        node_module.train()
        sa.train()
        epoch_cos_sim = []
        for i, (images, lowlight) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs_no}")
        ):
            images, lowlight = images.to(device), lowlight.to(device)

            # features, features_lowlight = feature_extractor.get_features_maps(images, lowlight)

            if args.batch_size == 1:
                # images, low_light = feature_extractor.get_features_maps(
                #     images, lowlight)
                images_feat, lowlight_feat = feature_extractor.get_features_maps(images, lowlight)
                
            else:
                # rgb_patches = []
                # xyz_patches = []
                images_feat_list, lowlight_feat_list = [], []
                # for i in range(images.shape[0]):
                #     rgb_patch, xyz_patch = feature_extractor.get_features_maps(images[i].unsqueeze(dim=0),
                #                                                                lowlight[i].unsqueeze(dim=0))
                #
                #     rgb_patches.append(rgb_patch)
                #     xyz_patches.append(xyz_patch)
                for j in range(images.shape[0]):
                    img_feat, low_feat = feature_extractor.get_features_maps(
                        images[j].unsqueeze(dim=0), lowlight[j].unsqueeze(dim=0)
                    )
                    low_feat, img_feat = node_module(low_feat, img_feat)
                    print(f'low_feat: {low_feat.shape}')
                    level_feat = feature_level_light(lowlight)
                    print(f'level_feat: {level_feat.shape}')
                    low_feat = torch.cat([low_feat, level_feat], dim=1)
                    # print(f'low_feat: {low_feat.shape}')
                    images_feat_list.append(img_feat)
                    lowlight_feat_list.append(low_feat)
                    # images_level_feat_list.append(level_feat)

                # images = torch.stack(rgb_patches, dim=0)
                # low_light = torch.stack(xyz_patches, dim=0)
                images_feat = torch.stack(images_feat_list, dim=0)
                lowlight_feat = torch.stack(lowlight_feat_list, dim=0)
                lowlight_feat = lowlight_feat.reshape(-1, 192 ,224, 224)
                images_feat = images_feat.reshape(-1, 64 ,224, 224)
                

                # images_level_feat = torch.stack(images_level_feat_list, dim=0)
                print(f'images_feat: {images_feat.shape}')
            # transfer_features = FAD_LLToClean(lowlight_feat)
            lowlight_feat = sa(lowlight_feat)
            # print(f'transfer_features: {transfer_features.shape}')
            # low_light_mask = (low_light.sum(axis=-1) == 0)
            mask = (lowlight_feat.sum(axis=-1) == 0)
            # loss = 1 - \
            #     metric(transfer_features[~low_light_mask],
            #            images[~low_light_mask]).mean()

            # loss = 1 - metric(transfer_features[~low_light_mask],low_light[~low_light_mask]).mean()
            # loss = 1 - metric(transfer_features[~low_light_mask], low_light[~low_light_mask]).mean()
            loss = 1 - metric(lowlight_feat[~mask], images_feat[~mask]).mean()

            # loss = 1 - metric(images, transfer_features).mean()
            #-------------------------------------------------
            # 1. la 2 loss, w-t, r-t
            # 2. using well light mask
            # 3. dung 2 mlp
            #---------------------------------------------------
            epoch_cos_sim.append(loss.item())
            if not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

        wandb.log(
            {
                "Epoch": epoch + 1,
                "Loss": np.mean(epoch_cos_sim),
            }
        )
        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)
        if (epoch + 1) % args.save_interval == 0:
            torch.save(
                FAD_LLToClean.state_dict(),
                f"{args.checkpoint_folder}/{args.class_name}/{model_name}",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Crossmodal Feature Networks (FADs) on a dataset."
    )

    parser.add_argument(
        "--dataset_path", default="data", type=str, help="Dataset path."
    )

    parser.add_argument(
        "--checkpoint_folder",
        default="checkpoints",
        type=str,
        help="Where to save the model checkpoints.",
    )

    parser.add_argument(
        "--class_name",
        default="medicine_pack",
        type=str,
        choices=[
            "small_battery",
            "screws_toys",
            "furniture",
            "tripod_plugs",
            "water_cups",
            "keys",
            "pens",
            "locks",
            "screwdrivers",
            "charging_cords",
            "pencil_cords",
            "water_cans",
            "pills",
            "locks",
            "medicine_pack",
            "small_bottles",
            "metal_plate",
            "usb_connector_board"
        ],
        help="Category name.",
    )

    parser.add_argument(
        "--epochs_no", default=10, type=int, help="Number of epochs to train the FADs."
    )

    parser.add_argument(
        "--batch_size",
        default=4,
        type=int,
        help="Batch dimension. Usually 16 is around the max.",
    )

    parser.add_argument(
        "--save_interval",
        default=5,
        type=int,
        help="Number of epochs to train the FADs.",
    )

    parser.add_argument("--unique_id", type=str, default="v2theta+",
                        help="A unique identifier for the checkpoint (e.g., experiment ID)")

    parser.add_argument("--person", default="DuongMinh" ,type=str,
                        help="Name or initials of the person saving the checkpoint")

    args = parser.parse_args()
    train(args)
