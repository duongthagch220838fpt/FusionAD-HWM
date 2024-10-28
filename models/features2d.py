import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from utils.metrics_utils import calculate_au_pro
from models.ad_models import FeatureExtractors


dino_backbone_name = 'vit_base_patch8_224.dino' # 224/8 -> 28 patches.
group_size = 128
num_group = 1024

class Multimodal2DFeatures(torch.nn.Module):
    def __init__(self, image_size = 224):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.deep_feature_extractor = FeatureExtractors(device = self.device, 
                                                 rgb_backbone_name = dino_backbone_name, 
                                                 group_size = group_size, num_group = num_group)

        self.deep_feature_extractor.to(self.device)

        self.image_size = image_size

        # * Applies a 2D adaptive average pooling over an input signal composed of several input planes. 
        # * The output is of size H x W, for any input size. The number of output features is equal to the number of input planes.
        self.resize = torch.nn.AdaptiveAvgPool2d((224, 224))
        
        self.average = torch.nn.AvgPool2d(kernel_size = 3, stride = 1) 

    def __call__(self, rgb, xyz):
        rgb, xyz = rgb.to(self.device), xyz.to(self.device)

        with torch.no_grad():
            # Extract feature maps from the 2D images
            rgb_feature_maps, xyz_feature_maps = self.deep_feature_extractor(rgb, xyz)

        return rgb_feature_maps,xyz_feature_maps

    def calculate_metrics(self):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        # Calculate ROC AUC scores and AU-PRO
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)

    def get_features_maps(self, rgb1, rgb2):
        # Ensure RGB inputs are on the correct device
        rgb1, rgb2 = rgb1.to(self.device), rgb2.to(self.device)

        # Extract feature maps from the 2 RGB images
        rgb_feature_maps1,rgb_feature_maps2 = self(rgb1, rgb2) 

        # Check if rgb_feature_maps1 is a single tensor or a list of tensors
        if isinstance(rgb_feature_maps1, list):
            rgb_patch1 = torch.cat(rgb_feature_maps1, 1)  # Concatenate if it's a list
        else:
            rgb_patch1 = rgb_feature_maps1  # Use it directly if it's a single tensor

        if isinstance(rgb_feature_maps2, list):
            rgb_patch2 = torch.cat(rgb_feature_maps2, 1)  # Concatenate if it's a list
        else:
            rgb_patch2 = rgb_feature_maps2  # Use it directly if it's a single tensor

        # Resize the feature maps to 224x224
        rgb_patch_resized1 = self.resize(rgb_patch1)
        rgb_patch_resized2 = self.resize(rgb_patch2)

        # Reshape to get the output as (C, H*W) where C is the number of channels
        rgb_patch_final1 = rgb_patch_resized1.view(rgb_patch_resized1.shape[1], -1).T
        rgb_patch_final2 = rgb_patch_resized2.view(rgb_patch_resized2.shape[1], -1).T

        return rgb_patch_final1, rgb_patch_final2
    

if __name__ == '__main__':
    model = Multimodal2DFeatures()
    x = torch.randn(1, 3, 224, 224)
    y = torch.randn(1, 3, 224, 224)
    out = model.get_features_maps(x, y)
    print(out[0].shape)