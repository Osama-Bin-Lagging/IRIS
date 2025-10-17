import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from components import ResidualBlock3D, DownsamplingBlock3D, ConvBlock3D

class Encoder3D(nn.Module):
    """
    3D UNet Encoder with multi-scale feature extraction
    Based on the Iris paper architecture with channel progression: [32, 32, 64, 128, 256, 512]
    """
    
    def __init__(self, in_channels: int = 1, base_channels: int = 32, num_blocks_per_stage: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels

        # FIXED: Correct UNet progression - 6 levels total including input level
        self.channels = [
            base_channels,      # 32  - Level 0
            base_channels,      # 32  - Level 1 (same resolution as input)
            base_channels * 2,  # 64  - Level 2 (first downsampling)
            base_channels * 4,  # 128 - Level 3
            base_channels * 8,  # 256 - Level 4
            base_channels * 16, # 512 - Level 5 (bottleneck)
        ]

        # Initial convolution
        self.stem = ConvBlock3D(in_channels, self.channels[0], kernel_size=3, padding=1)
        
        # Encoder stages
        self.stages = nn.ModuleList()
        
        # Stage 0: No downsampling, just process features at input resolution
        stage0 = nn.Sequential()
        for i in range(num_blocks_per_stage):
            stage0.add_module(f'block_{i}', 
                            ResidualBlock3D(self.channels[0], self.channels[1]))
        self.stages.append(stage0)
        
        # FIXED: Stages 1-4 (4 downsampling stages, not 5)
        # This creates 5 stages total (0-4), matching the 6 channel levels
        for i in range(1, len(self.channels) - 1):  # FIXED: -1 to prevent index error
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            stage = DownsamplingBlock3D(in_ch, out_ch, num_blocks_per_stage)
            self.stages.append(stage)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass returning multi-scale features
        
        Args:
            x: Input tensor of shape (B, C, D, H, W)
            
        Returns:
            List of feature maps at different scales
        """
        features = []
        
        # Stem convolution
        x = self.stem(x)
        features.append(x)
        
        # Pass through encoder stages
        for i, stage in enumerate(self.stages):
            x = stage(x)
            features.append(x)
        
        return features
    
    def get_feature_channels(self) -> List[int]:
        """Get the number of channels for each feature level"""
        return self.channels
    
    def get_feature_scales(self) -> List[int]:
        """Get the downsampling scales for each feature level"""
        return [1, 1, 2, 4, 8, 16]  # Matches the 6 levels

class EncoderWithSkipConnections(nn.Module):
    """
    Enhanced 3D Encoder with skip connections for better gradient flow
    """
    
    def __init__(self,
                 in_channels: int = 1,
                 base_channels: int = 32,
                 num_blocks_per_stage: int = 2,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.encoder = Encoder3D(in_channels, base_channels, num_blocks_per_stage)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
        # Additional processing layers for skip connections
        channels = self.encoder.get_feature_channels()
        self.skip_convs = nn.ModuleList([
            ConvBlock3D(ch, ch, kernel_size=1, padding=0) for ch in channels
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with skip connections
        
        Returns:
            Tuple of (main_features, skip_features)
        """
        # Get multi-scale features
        main_features = self.encoder(x)
        
        # Process skip connections
        skip_features = []
        for i, (feat, skip_conv) in enumerate(zip(main_features, self.skip_convs)):
            skip_feat = skip_conv(feat)
            skip_feat = self.dropout(skip_feat)
            skip_features.append(skip_feat)
        
        return main_features, skip_features

# Test function to verify encoder architecture
def test_encoder():
    """Test the 3D encoder implementation"""
    print("Testing 3D Encoder...")
    
    # Create sample input
    batch_size = 2
    input_size = (128, 128, 128)  # D, H, W
    x = torch.randn(batch_size, 1, *input_size)
    
    # Initialize encoder
    encoder = Encoder3D(in_channels=1, base_channels=32)
    
    # Forward pass
    with torch.no_grad():
        features = encoder(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Number of feature levels: {len(features)}")
    
    for i, feat in enumerate(features):
        scale = encoder.get_feature_scales()[i]
        channels = encoder.get_feature_channels()[i]
        print(f"Level {i}: {feat.shape}, channels={channels}, scale=1/{scale}")
    
    # Test enhanced encoder
    print("\nTesting Enhanced Encoder with Skip Connections...")
    enhanced_encoder = EncoderWithSkipConnections(in_channels=1, base_channels=32)
    
    with torch.no_grad():
        main_features, skip_features = enhanced_encoder(x)
    
    print(f"Main features: {len(main_features)} levels")
    print(f"Skip features: {len(skip_features)} levels")
    
    return encoder, enhanced_encoder

if __name__ == "__main__":
    test_encoder()