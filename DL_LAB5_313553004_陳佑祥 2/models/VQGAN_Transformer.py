import torch 
import torch.nn as nn
from torch import Tensor
import yaml
import os
import math
import numpy as np
from .VQGAN import VQGAN
from .Transformer import BidirectionalTransformer
import torch.nn.functional as F
from typing import Tuple

# This is correct !! 

#TODO2 step1: design the MaskGIT model
class MaskGit(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.vqgan = self.load_vqgan(configs['VQ_Configs'])
    
        self.num_image_tokens = configs['num_image_tokens']
        self.mask_token_id = configs['num_codebook_vectors']
        self.choice_temperature = configs['choice_temperature']
        self.gamma = self.gamma_func(configs['gamma_type'])
        self.transformer = BidirectionalTransformer(configs['Transformer_param'])

    def load_transformer_checkpoint(self, load_ckpt_path):
        # Directly load the transformer state dict
        transformer_state_dict = torch.load(load_ckpt_path)
        self.transformer.load_state_dict(transformer_state_dict)


    @staticmethod
    def load_vqgan(configs):
        cfg = yaml.safe_load(open(configs['VQ_config_path'], 'r'))
        model = VQGAN(cfg['model_param'])
        model.load_state_dict(torch.load(configs['VQ_CKPT_path']), strict=True) 
        model = model.eval()
        return model
    
##TODO2 step1-1: input x fed to vqgan encoder to get the latent and zq
    @torch.no_grad()
    def encode_to_z(self, x):
        # Use the VQGAN's encode function to get the codebook mapping and indices
        codebook_mapping, codebook_indices, _ = self.vqgan.encode(x)
        
        # Flatten the codebook mapping to prepare it for the transformer
        codebook_indices_flat = codebook_indices.contiguous().view(codebook_mapping.size(0), -1)

        return codebook_mapping, codebook_indices_flat

    
##TODO2 step1-2:    
    def gamma_func(self, mode="cosine"):
        """Generates a mask rate by scheduling mask functions R.

        Given a ratio in [0, 1), we generate a masking ratio from (0, 1]. 
        During training, the input ratio is uniformly sampled; 
        during inference, the input ratio is based on the step number divided by the total iteration number: t/T.
        Based on experiements, we find that masking more in training helps.
        
        ratio:   The uniformly sampled ratio [0, 1) as input.
        Returns: The mask rate (float).

        """
        if mode == "linear":
            return lambda t: 1 - t  # linear scheduling
        elif mode == "cosine":
            return lambda t: 0.5 * (1 + math.cos(math.pi * t))  # cosine scheduling
        elif mode == "square":
            return lambda t: 1 - (t ** 2)  # square scheduling
        # 也可implement更多種mask方式
        else:
            raise NotImplementedError

##TODO2 step1-3:            
    def forward(self, x):
        # Step 1: Get the latent representation (codebook_mapping) and the flattened codebook indices
        codebook_mapping, original_indices = self.encode_to_z(x)
        
        # Step 2: Create a mask token tensor filled with the mask token ID
        mask_token_tensor = torch.full_like(original_indices, self.mask_token_id)

        # Step 3: Generate a random ratio between 0 and 1
        random_ratio = torch.rand(1).item()  # Random number between 0 and 1

        # Step 4: Generate a random binary mask based on the random ratio
        random_mask = torch.bernoulli(random_ratio * torch.ones_like(original_indices)).bool()
        
        # Step 5: Apply the mask: replace masked tokens with mask token ID, keep others as original
        masked_indices = torch.where(random_mask, mask_token_tensor, original_indices)
        
        # Step 6: Pass the masked indices through the transformer to get logits
        logits = self.transformer(masked_indices)
        
        return logits, original_indices

    
##TODO3 step1-1: define one iteration decoding   
    @torch.no_grad()
    def inpainting(self, ratio: float, z_indices: Tensor, mask: Tensor, mask_num: int):
        device = mask.device

        # Apply the mask to z_indices, replacing masked positions with mask_token_id
        masked_indices = self._apply_mask(z_indices, mask)

        # Generate logits and corresponding probabilities from the transformer
        probability = self._get_probabilities(masked_indices)

        # Predict z_indices and compute confidence based on the temperature
        z_indices_predict, confidence = self._predict_with_confidence(probability, z_indices, mask, ratio, device)

        # Update the mask for the next iteration, now passing `ratio`
        mask_bc = self._update_mask(confidence, mask, mask_num, ratio)

        return z_indices_predict, mask_bc


    def _apply_mask(self, z_indices: Tensor, mask: Tensor) -> Tensor:
        """Apply the mask to the z_indices."""
        return torch.where(mask, torch.full_like(z_indices, self.mask_token_id), z_indices)

    def _get_probabilities(self, masked_indices: Tensor) -> Tensor:
        """Pass the masked indices through the transformer and apply softmax to get probabilities."""
        logits = self.transformer(masked_indices)
        return torch.nn.functional.softmax(logits, dim=-1)

    def _predict_with_confidence(self, probability: Tensor, z_indices: Tensor, mask: Tensor, ratio: float, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Predict the z_indices and compute the confidence with temperature annealing."""
        z_indices_predict_prob, z_indices_predict = probability.max(dim=-1)
        z_indices_predict = torch.where(mask, z_indices_predict, z_indices)

        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(z_indices_predict_prob.shape).to(device)
        temperature = self.choice_temperature * (1 - ratio)
        confidence = z_indices_predict_prob + temperature * gumbel_noise

        confidence.masked_fill_(~mask, float('inf'))
        
        return z_indices_predict, confidence

    def _update_mask(self, confidence: Tensor, mask: Tensor, mask_num: int, ratio: float) -> Tensor:
        """Update the mask by selecting the top-k smallest confidence values."""
        n = math.ceil(self.gamma(ratio) * mask_num)
        _, idx = confidence.topk(n, dim=-1, largest=False)

        # Initialize a new mask based on the top-k indices
        new_mask_bc = torch.zeros_like(mask, dtype=torch.bool)
        new_mask_bc.scatter_(dim=1, index=idx, value=True)

        # Retain the existing masked positions and apply the new mask update
        mask_bc = mask & new_mask_bc
        
        return mask_bc



    
__MODEL_TYPE__ = {
    "MaskGit": MaskGit
}
    


        
