import argparse
import os
import torch
from torchvision import transforms as T
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.features import MultimodalFeatures
# from models.dataset import get_data_loader
from models.feature_transfer_nets import FeatureProjectionMLP, FeatureProjectionMLP_big
from models.ad_models import FeatureExtractors
from models.features2d import Multimodal2DFeatures
from utils.metrics_utils import calculate_au_pro
from sklearn.metrics import roc_auc_score
from dataset2D import SquarePad, TestDataset


def set_seeds(sid=42):
    np.random.seed(sid)

    torch.manual_seed(sid)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(sid)
        torch.cuda.manual_seed_all(sid)


def infer(args):
    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f"{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs"

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
    # test_loader = get_dataloader(
    #     "Anomaly", class_name=args.class_name, img_size=224, dataset_path=args.dataset_path
    # )
    test_dataset = TestDataset('data/Anomaly', 'data/gt', transform=common, low_light_transform=common)

    test_loader = DataLoader(dataset = test_dataset, batch_size = 1, num_workers = 16)
    # Feature extractors.
    feature_extractor = Multimodal2DFeatures(image_size = 224)
    

    # Model instantiation.
    FAD_LLToClean = FeatureProjectionMLP(in_features=768, out_features=768)
    

    FAD_LLToClean.to(device)
    feature_extractor.to(device)
    FAD_LLToClean.load_state_dict(torch.load("/home/tanpx/PycharmProjects/FusionADver2/FusionAD-DuongMinh/checkpoints/Bowl/FAD_LLToClean_Bowl_10ep_4bs.pth"))

    FAD_LLToClean.eval()
    feature_extractor.eval()

    # Metrics.
    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    # Use box filters to approximate gaussian blur (https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf).
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device = device)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device = device)/(w_u**2)

    predictions, gts = [], []
    pixel_labels = []
    image_preds, pixel_preds = [], []


    # Inference.
    # Assuming FeatureExtractor is for 2D images, and we use one FeatureProjectionMLP for projection.
# ------------ [Testing Loop] ------------ #

# * Return (img1, img2), gt[:1], label, img_path where img1 and img2 are both 2D images.
    for img1, img2, gt in  tqdm(test_loader, desc=f'Extracting feature from class: {args.class_name}.'):

        # Move data to the GPU (or whatever device is being used)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            # Extract features from both 2D images
            img1_features,img2_features = feature_extractor.get_features_maps(img1, img2)  # Features from img2 (e.g., low-light image)
            
            # Project features from img2 into the same space as img1 using the FeatureProjectionMLP
            projected_img2_features = FAD_LLToClean(img2_features)  # FeatureProjectionMLP now projects between 2D features
        
            # Mask invalid features (if necessary)
            feature_mask = (img2_features.sum(axis=-1) == 0)  # Mask for img2 features that are all zeros.
            # feature_mask = (img2_features.sum(dim=-1) == 0).unsqueeze(-1)  # Shape: (1, 785, 1)


            # Cosine distance between img1 features and projected img2 features
            cos_img1 = (torch.nn.functional.normalize(img1_features, dim=1) -
                        torch.nn.functional.normalize(img1_features, dim=1)).pow(2).sum(1).sqrt()
            cos_img2 = (torch.nn.functional.normalize(projected_img2_features, dim=1) -
                        torch.nn.functional.normalize(img2_features, dim=1)).pow(2).sum(1).sqrt()



            # print(cos_img1.shape)
            # Apply the mask to ignore zeroed out features
            cos_img1[feature_mask] = 0.
            # cos_img1[feature_mask.squeeze(-1)] = 0

            cos_img1 = cos_img1.reshape(224, 224)
            # cos_img1 = cos_img1.view(1, 1, 155, 157)  # Reshape for interpolation
            # cos_img1 = torch.nn.functional.interpolate(cos_img1, size=(224, 224), mode="bilinear", align_corners=False)
            # cos_img1 = cos_img1.squeeze()  # Shape: (224, 224)


            cos_img2[feature_mask] = 0.
            cos_img2 = cos_img2.reshape(224, 224)

            # cos_img2 = cos_img2.view(1, 1, 155, 157)  # Reshape for interpolation
            # cos_img2 = torch.nn.functional.interpolate(cos_img2, size=(224, 224), mode="bilinear", align_corners=False)
            # cos_img2 = cos_img2.squeeze()  # Shape: (224, 224)

            # Combine the cosine distances from both feature sets
            cos_comb = (cos_img2)
            # print("Cos_comb")
            # print(cos_comb.shape)
            # print("Feature_mask")
            # print(feature_mask.shape)
            cos_comb.reshape(-1)[feature_mask] = 0.

            # Apply smoothing (similarly as before) using conv2d
            cos_comb = cos_comb.reshape(1, 1, 224, 224)
            
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            
            cos_comb = cos_comb.reshape(224, 224)
            
            # Prediction and ground-truth accumulation.
            gts.append(gt.squeeze().cpu().detach().numpy())  # (224, 224)
            predictions.append((cos_comb / (cos_comb[cos_comb != 0].mean())).cpu().detach().numpy())  # (224, 224)
            # print("cos_comb not shape")
            # print(cos_comb)
            # GTs
            # image_labels.append()  # (1,)
            pixel_labels.extend(gt.flatten().cpu().detach().numpy())  # (50176,)
            # print("PIXEL LABEL: ")
            # print(pixel_labels[0])
            # Predictions
            # image_preds.append((cos_comb / torch.sqrt(cos_comb[cos_comb != 0].mean())).cpu().detach().numpy().max())  # single number
            pixel_preds.extend((cos_comb / torch.sqrt(cos_comb.mean())).flatten().cpu().detach().numpy())  # (224, 224)
            # print("pixel_preds")
            # print(pixel_preds[0])

            # Calculate AD&S metrics.
            au_pros, _ = calculate_au_pro(gts, predictions)

            pixel_rocauc = roc_auc_score(np.stack(pixel_labels), np.stack(pixel_preds))
            # valid_indices = ~np.isnan(pixel_labels) & ~np.isnan(pixel_preds)
            # pixel_labels_clean = np.array(pixel_labels)[valid_indices]
            # pixel_preds_clean = np.array(pixel_preds)[valid_indices]
            # pixel_rocauc = roc_auc_score(pixel_labels_clean, pixel_preds_clean)
            # image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

            result_file_name = f'{args.quantitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'

            title_string = f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs'
            header_string = 'AUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC'
            results_string = f'{au_pros[0]:.3f} & {au_pros[1]:.3f} & {au_pros[2]:.3f} & {au_pros[3]:.3f} & {pixel_rocauc:.3f} '

            if not os.path.exists(args.quantitative_folder):
                os.makedirs(args.quantitative_folder)

            with open(result_file_name, "w") as markdown_file:
                markdown_file.write(title_string + '\n' + header_string + '\n' + results_string)

            # Print AD&S metrics.
            print(title_string)
            print("AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% | P-AUROC")
            print(
                f'  {au_pros[0]:.3f}   |   {au_pros[1]:.3f}   |   {au_pros[2]:.3f}  |   {au_pros[3]:.3f}  |   {pixel_rocauc:.3f} |',
                end='\n')
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make inference with Crossmodal Feature Networks (FADs) on a dataset.')

    parser.add_argument('--dataset_path', default='./datasets/mvtec3d', type=str,
                        help='Dataset path.')

    parser.add_argument('--class_name', default=None, type=str,
                        choices=["bagel", "cable_gland", "carrot", "cookie", "dowel", "foam", "peach", "potato", "rope",
                                 "tire",
                                 'CandyCane', 'ChocolateCookie', 'ChocolatePraline', 'Confetto', 'GummyBear',
                                 'HazelnutTruffle', 'LicoriceSandwich', 'Lollipop', 'Marshmallow', 'PeppermintCandy'],
                        help='Category name.')

    parser.add_argument('--checkpoint_folder', default='./checkpoints/checkpoints_FAD_mvtec', type=str,
                        help='Path to the folder containing FADs checkpoints.')

    parser.add_argument('--qualitative_folder', default='./results/qualitatives_mvtec', type=str,
                        help='Path to the folder in which to save the qualitatives.')

    parser.add_argument('--quantitative_folder', default='./results/quantitatives_mvtec', type=str,
                        help='Path to the folder in which to save the quantitatives.')

    parser.add_argument('--epochs_no', default=50, type=int,
                        help='Number of epochs to train the FADs.')

    parser.add_argument('--batch_size', default=4, type=int,
                        help='Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--visualize_plot', default=False, action='store_true',
                        help='Whether to show plot or not.')

    parser.add_argument('--produce_qualitatives', default=False, action='store_true',
                        help='Whether to produce qualitatives or not.')

    args = parser.parse_args()

    infer(args)
                