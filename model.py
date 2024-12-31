import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 1. Enhanced Edge Detection Module
class EnhancedEdgeDetection(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(EnhancedEdgeDetection, self).__init__()
        # A deeper trainable convolutional block for learning better edge features
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        edges = self.conv3(x)
        return edges

# 2. Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# 3. Channel-wise Attention Module (Squeeze-and-Excitation)
class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.avg_pool(x)
        attention = self.fc1(attention)
        attention = self.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        return x * attention

# 4. Final Refinement Block with Residual Learning
class EdgeAwareRefinementBlock(nn.Module):
    def __init__(self, in_channels=21, out_channels=1, reduction=16):
        super(EdgeAwareRefinementBlock, self).__init__()
        # Initialize sub-modules
        self.edge_detection = EnhancedEdgeDetection(in_channels=in_channels, out_channels=out_channels)
        self.channel_attention = ChannelAttentionModule(in_channels=in_channels, reduction=reduction)
        self.spatial_attention = SpatialAttention()
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # Apply edge detection module
        edge_output = self.edge_detection(x)

        # Apply channel and spatial attention refinement
        attention_output = self.channel_attention(x)
        spatial_attention_output = self.spatial_attention(attention_output)
        attention_output = attention_output * spatial_attention_output

        # Residual connection with edge output
        refined_output = edge_output + attention_output

        # Final single-channel output
        final_output = self.final_conv(refined_output)
        final_output = torch.sigmoid(final_output)
        return final_output

# Using Pretrained DeepLabV3 (ResNet50 backbone)
class ERNetModel(nn.Module):
    def __init__(self, pretrained=True):
        super(ERNetModel, self).__init__()
        
        # Load the pretrained deeplabv3_resnet50 model
        if pretrained: self.deeplabv3 = models.segmentation.deeplabv3_resnet50(weights="DeepLabV3_ResNet50_Weights.DEFAULT", weights_backbone="ResNet50_Weights.DEFAULT")
        else: self.deeplabv3 = models.segmentation.deeplabv3_resnet50()

        self.refinement_block = EdgeAwareRefinementBlock(in_channels=21, out_channels=1)
        
    def forward(self, x):
        return self.refinement_block(self.deeplabv3(x)['out'])

# Initialize the refinement block and the full model
if __name__ == "__main__":
    model = ERNetModel()
    print(model)
