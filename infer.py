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
    path_gt = args.dataset_path + '/' + args.class_name + "/gt"
    path_anomaly = args.dataset_path + '/' +args.class_name + "/anomaly"
    test_dataset = TestDataset(
        path_anomaly, path_gt, transform=common, low_light_transform=common
    )

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1, num_workers=16)
    # Feature extractors.
    feature_extractor = Multimodal2DFeatures(image_size=224)

    # Model instantiation.
    FAD_LLToClean = FeatureProjectionMLP(in_features=768, out_features=768)

    FAD_LLToClean.to(device)
    feature_extractor.to(device)
    model_name = f"FAD_LLToClean{args.person}{args.unique_id}.pth"
    FAD_LLToClean.load_state_dict(
        torch.load(
            f"{args.checkpoint_folder}/{args.class_name}/{model_name}",
        )
    )
    # f"{args.checkpoint_folder}/{args.class_name}/FAD_LLToClean_{
    # args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth",
    FAD_LLToClean.eval()
    feature_extractor.eval()

    # Metrics.
    metric = torch.nn.CosineSimilarity(dim=-1, eps=1e-06)

    # Use box filters to approximate gaussian blur (https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf).
    # w_l, w_u = 5, 7
    # pad_l, pad_u = 2, 3
    # weight_l = torch.ones(1, 1, w_l, w_l, device = device)/(w_l**2)
    # weight_u = torch.ones(1, 1, w_u, w_u, device = device)/(w_u**2)

    predictions, gts = [], []
    pixel_labels = []
    image_preds, pixel_preds = [], []

    # Inference.
    # Assuming FeatureExtractor is for 2D images, and we use one FeatureProjectionMLP for projection.
    # ------------ [Testing Loop] ------------ #

    # * Return (img1, img2), gt[:1], label, img_path where img1 and img2 are both 2D images.
    for img1, img2, gt, path_image_low, path_image_high in tqdm(
        test_loader, desc=f"Extracting feature from class: {args.class_name}."
    ):
        # Move old_data to the GPU (or whatever device is being used)
        img1, img2 = img1.to(device), img2.to(device)

        with torch.no_grad():
            # Extract features from both 2D images
            img1_features, img2_features = feature_extractor.get_features_maps(
                img1, img2
            )  # Features from img2 (e.g., low-light image)

            # Project features from img2 into the same space as img1 using the FeatureProjectionMLP
            # FeatureProjectionMLP now projects between 2D features
            projected_img2_features = FAD_LLToClean(img2_features)

            # Mask invalid features (if necessary)
            # Mask for img2 features that are all zeros.
            feature_mask = img2_features.sum(axis=-1) == 0
            # feature_mask = (img2_features.sum(dim=-1) == 0).unsqueeze(-1)  # Shape: (1, 785, 1)
            cos_img1 = (torch.nn.functional.normalize(img1_features, dim=1) -
                        torch.nn.functional.normalize(img2_features, dim=1)).pow(2).sum(1).sqrt()
            # Cosine distance between img1 features and projected img2 features
            cos_img2 = (
                (
                    torch.nn.functional.normalize(
                        projected_img2_features, dim=1)
                    - torch.nn.functional.normalize(img1_features, dim=1)
                )
                .pow(2)
                .sum(1)
                .sqrt()
            )

            cos_img2[feature_mask] = 0.0
            cos_img2 = cos_img2.reshape(224, 224)



            # cos_img2 = cos_img2.view(1, 1, 155, 157)  # Reshape for interpolation
            # cos_img2 = torch.nn.functional.interpolate(cos_img2, size=(224, 224), mode="bilinear", align_corners=False)
            # cos_img2 = cos_img2.squeeze()  # Shape: (224, 224)

            # Combine the cosine distances from both feature sets

            # print("Cos_comb")
            # print(cos_comb.shape)
            # print("Feature_mask")
            # print(feature_mask.shape)
            # cos_comb.reshape(-1)[feature_mask] = 0.

            # Apply smoothing (similarly as before) using conv2d
            # cos_comb = cos_comb.reshape(1, 1, 224, 224)

            # cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            # cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            # cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            # cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)
            # cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_l, weight=weight_l)

            # cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            # cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)
            # cos_comb = torch.nn.functional.conv2d(input=cos_comb, padding=pad_u, weight=weight_u)

            # cos_comb = cos_comb.reshape(224, 224)

            # Prediction and ground-truth accumulation.
            gts.append(gt.squeeze().cpu().detach().numpy())  # (224, 224)
            predictions.append(
                # (224, 224)
                (cos_img2 / (cos_img2[cos_img2 != 0].mean())
                 ).cpu().detach().numpy()
            )
            # print("cos_comb not shape")
            # print(cos_comb)
            # GTs
            # image_labels.append()  # (1,)
            pixel_labels.extend(
                gt.flatten().cpu().detach().numpy())  # (50176,)
            # print("PIXEL LABEL: ")
            # print(pixel_labels[0])
            # Predictions
            # image_preds.append((cos_comb / torch.sqrt(cos_comb[cos_comb != 0].mean())).cpu().detach().numpy().max())  # single number
            pixel_preds.extend(
                (
                    cos_img2 / torch.sqrt(cos_img2.mean())
                    # (224, 224)
                )
                .flatten()
                .cpu()
                .detach()
                .numpy()
            )
            # print("pixel_preds")
            # print(pixel_preds[0])
            if args.produce_qualitatives:
                defect_class_str = path_image_low[0].split("/")[-4]
                image_name_str = path_image_low[0].split("/")[-1]

                save_path = f'{args.qualitative_folder}/{args.class_name}_{args.epochs_no}bs/{defect_class_str}'
                

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                fig, axs = plt.subplots(2, 2, figsize=(7, 7))

                denormalize = T.Compose(
                    [
                        T.Normalize(
                            mean=[0.0, 0.0, 0.0], std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
                        ),
                        T.Normalize(
                            mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
                    ]
                )

                rgb = denormalize(img2)

                os.path.join(save_path, image_name_str)

                axs[0, 0].imshow(rgb.squeeze().permute(
                    1, 2, 0).cpu().detach().numpy())
                axs[0, 0].set_title("RGB")

                axs[0, 1].imshow(gt.squeeze().cpu().detach().numpy())
                axs[0, 1].set_title("Ground-truth")

                axs[1, 1].imshow(
                    cos_img2.cpu().detach().numpy(), cmap=plt.cm.jet)
                axs[1, 1].set_title("2D Cosine Similarity")

                # axs[1, 2].imshow(
                #     cos_img2.cpu().detach().numpy(), cmap=plt.cm.jet)
                # axs[1, 2].set_title('Combined Cosine Similarity')

                # Remove ticks and labels from all subplots
                for ax in axs.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                # Adjust the layout and spacing
                plt.tight_layout()

                plt.savefig(os.path.join(save_path, image_name_str), dpi=256)

                if args.visualize_plot:
                    plt.show()
                    plt.pause(5)
                    plt.close()

                    # Calculate AD&S metrics.
            au_pros, _ = calculate_au_pro(gts, predictions)

            pixel_rocauc = roc_auc_score(
                np.stack(pixel_labels), np.stack(pixel_preds))
            # valid_indices = ~np.isnan(pixel_labels) & ~np.isnan(pixel_preds)
            # pixel_labels_clean = np.array(pixel_labels)[valid_indices]
            # pixel_preds_clean = np.array(pixel_preds)[valid_indices]
            # pixel_rocauc = roc_auc_score(pixel_labels_clean, pixel_preds_clean)
            # image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

            result_file_name = f"{args.quantitative_folder}/{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs{args.unique_id}.md"

            title_string = f"Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs"
            header_string = "AUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC"
            results_string = f"{au_pros[0]:.3f} & {au_pros[1]:.3f} & {au_pros[2]:.3f} & {au_pros[3]:.3f} & {pixel_rocauc:.3f}"

            if not os.path.exists(args.quantitative_folder):
                os.makedirs(args.quantitative_folder)

            with open(result_file_name, "w") as markdown_file:
                markdown_file.write(
                    title_string + "\n" + header_string + "\n" + results_string
                )

            # Print AD&S metrics.
            print(title_string)
            print("AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% | P-AUROC")
            print(f"  {au_pros[0]:.3f}   |   {au_pros[1]:.3f}   |   {au_pros[2]:.3f}  |   {au_pros[3]:.3f}  |   {pixel_rocauc:.3f} |",end="\n",)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make inference with Crossmodal Feature Networks (FADs) on a dataset."
    )

    parser.add_argument(
        "--dataset_path", default="data", type=str, help="Dataset path."
    )

    parser.add_argument(
        "--qualitative_folder",
        default="./results/qualitatives",
        type=str,
        help="Path to the folder in which to save the qualitatives.",
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
            "usb_connector_board",
        ],
        help="Category name.",
    )

    parser.add_argument(
        "--checkpoint_folder",
        default="./checkpoints",
        type=str,
        help="Path to the folder containing FADs checkpoints.",
    )

    parser.add_argument(
        "--quantitative_folder",
        default="./results/quantitatives",
        type=str,
        help="Path to the folder in which to save the quantitatives.",
    )

    parser.add_argument(
        "--epochs_no", default=10, type=int, help="Number of epochs to train the FADs."
    )

    parser.add_argument(
        "--batch_size_train",
        default=4,
        type=int,
        help="Batch dimension. Usually 16 is around the max.",
    )

    parser.add_argument(
        "--batch_size",
        default=1,
        type=int,
        help="Batch dimension. Usually 16 is around the max.",
    )

    parser.add_argument(
        "--visualize_plot",
        default=False,
        action="store_true",
        help="Whether to show plot or not.",
    )

    parser.add_argument(
        "--produce_qualitatives",
        default=True,
        action="store_true",
        help="Whether to produce qualitatives or not.",
    )

    parser.add_argument("--unique_id", type=str, default="test+",
                        help="A unique identifier for the checkpoint (e.g., experiment ID)")

    parser.add_argument("--person", default="DuongMinh" ,type=str,
                        help="Name or initials of the person saving the checkpoint")

    args = parser.parse_args()

    infer(args)
