import torch
import numpy as np

from sklearn.metrics import roc_auc_score
from utils.metrics_utils import calculate_au_pro
from models.ad_models import FeatureExtractors

from torch.profiler import profile, record_function, ProfilerActivity

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
        # print(rgb1.shape)
        # Extract feature maps from the 2 RGB images
        rgb_feature_maps1,rgb_feature_maps2 = self(rgb1, rgb2) 
        # print(rgb_feature_maps1.shape)
        # Check if rgb_feature_maps1 is a single tensor or a list of tensors
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
        # print("Shape of rgb_patch1 before resize:", rgb_patch1.shape)  # Expected (1, 785, 768) or similar

        
        # Step 2: Interpolate to (224, 224)
        rgb_patch_upsample1 = torch.nn.functional.interpolate(rgb_patch1, size=(224, 224), mode='bilinear', align_corners=False)
        # rgb_patch_upsample1 = torch.nn.functional.interpolate(rgb_patch1, size=(224, 224), mode='bilinear', align_corners=False)
        rgb_patch_upsample2 = torch.nn.functional.interpolate(rgb_patch2, size=(224, 224), mode='bilinear', align_corners=False)

        # print("Shape of rgb_patch_upsample1:", rgb_patch_upsample1.shape)  # Expect (1, 768, 224, 224
        # Step 3: Reshape to (H*W, C) which is (50176, 768)
        rgb_patch_final1 =  rgb_patch_upsample1.reshape(rgb_patch1.shape[1], -1).T
        # Step 3: Reshape to (H*W, C) which is (50176, 768)
        rgb_patch_final2 =  rgb_patch_upsample2.reshape(rgb_patch2.shape[1], -1).T
        # Print final shape for verification
        # print("Shape of rgb_patch_final1:", rgb_patch_final1.shape)  # Expect (50176, 768)

        return rgb_patch_final1, rgb_patch_final2

        
    

if __name__ == '__main__':


    model = Multimodal2DFeatures()
    rgb1 = torch.randn(1, 3, 224, 224)
    rgb1 = torch.randn(1, 3, 224, 224)
    rgb_patch_final1, rgb_patch_final2 = model.get_features_maps(rgb1, rgb1)

    # Check if the output is still in (224, 224) format and flatten it if necessary
    # rgb_patch_final1 should be of shape (50176, C) where C is the channel dimension of the final feature map
    if rgb_patch_final1.shape[-2:] == (224, 224):
        rgb_patch_final1 = rgb_patch_final1.view(-1, rgb_patch_final1.shape[1])
        rgb_patch_final2 = rgb_patch_final2.view(-1, rgb_patch_final2.shape[1])

    # Output the shape to confirm correctness
    # print("Shape of rgb_patch_final1:", rgb_patch_final1.shape)
    # print("Shape of rgb_patch_final2:", rgb_patch_final2.shape)
    #
    # model = FeatureExtractors(device='CPU')
    # model.eval()
    # inputs = torch.rand(1, 3, 512, 512)
    # with profile(activities=[ProfilerActivity.CPU],
    #              profile_memory=True, record_shapes=True) as prof:
    #     model(inputs, inputs)
    #
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=30))