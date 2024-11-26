import torch
import torch.nn as nn
from diffusers import UNet2DModel

class CustomDDPM(nn.Module):
    def __init__(self, num_classes=24, model_dim=512):
        super(CustomDDPM, self).__init__()
        
        self.embedding_dim = model_dim // 4
        # we can change for different embedding for labels
        self.embedding_layer = nn.Linear(num_classes, model_dim)
        
        self.unet_model = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=[
                self.embedding_dim,
                self.embedding_dim,
                self.embedding_dim * 2,
                self.embedding_dim * 2,
                self.embedding_dim * 4,
                self.embedding_dim * 4
            ],
            down_block_types=[
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D"
            ],
            up_block_types=[
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D"
            ],
            class_embed_type="identity"
        )
    
    def forward(self, input_tensor, time_step, labels):
        embedded_class = self.embedding_layer(labels)
        output = self.unet_model(input_tensor, time_step, embedded_class)
        return output.sample
