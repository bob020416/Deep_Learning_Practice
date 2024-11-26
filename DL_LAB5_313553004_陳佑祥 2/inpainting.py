import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import argparse
from utils import LoadTestData, LoadMaskData
from torch.utils.data import Dataset,DataLoader
from torchvision import utils as vutils
import os
from models import MaskGit as VQGANTransformer
import yaml
import torch.nn.functional as F

class MaskGIT:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.model.load_transformer_checkpoint(args.load_transformer_ckpt_path)
        self.model.eval()
        self.total_iter=args.total_iter
        self.mask_func=args.mask_func
        self.sweet_spot=args.sweet_spot
        self.device=args.device
        self.prepare()

    @staticmethod
    def prepare():
        os.makedirs("test_results", exist_ok=True)
        os.makedirs("mask_scheduling", exist_ok=True)
        os.makedirs("imga", exist_ok=True)

##TODO3 step1-1: total iteration decoding  
#mask_b: iteration decoding initial mask, where mask_b is true means mask
    def inpainting(self, image, mask_b, i): 
        maska = torch.zeros(self.total_iter, 3, 16, 16)  # Save all iterations of masks in latent domain
        imga = torch.zeros(self.total_iter + 1, 3, 64, 64)  # Save all iterations of decoded images
        mean = torch.tensor([0.4868, 0.4341, 0.3844], device=self.device).view(3, 1, 1)
        std = torch.tensor([0.2620, 0.2527, 0.2543], device=self.device).view(3, 1, 1)
        ori = (image[0] * std) + mean
        imga[0] = ori  # The first image is the ground truth of the masked image

        self.model.eval()
        with torch.no_grad():
            # Encode Z
            encode_to_latent = lambda img: self.model.encode_to_z(img)[1]
            z_indices_predict = encode_to_latent(image[0].unsqueeze(0))

            mask_bc = mask_b.to(self.device)
            mask_num = mask_b.sum() # total number of mask token 

            for step in range(self.total_iter):
                if step == self.sweet_spot:
                    break
                
                # another mask ratio testing here
                if self.mask_func == 'linear':
                    ratio = (step + 1) / self.total_iter
                elif self.mask_func == 'cosine':
                    import math 
                    ratio = 0.5 * (1 + math.cos(math.pi * (1 - (step + 1) / self.total_iter)))


                elif self.mask_func == 'square':
                    ratio = ((step + 1) / self.total_iter) ** 2
                else:
                    raise ValueError(f"Unknown mask_func: {self.mask_func}")

                import torch.nn.functional as F

                # Perform inpainting
                z_indices_predict, mask_bc = self.model.inpainting(ratio, z_indices_predict, mask_bc, mask_num)

                
                # Debug: Check the mask sum
                print(f"Step {step + 1}: Mask sum (number of masked tokens) = {mask_bc.sum().item()}")

                # Update the mask visualization and decoded image storage
                mask_i = mask_bc.view(1, 16, 16)
                mask_image = torch.ones(3, 16, 16)
                indices = torch.nonzero(mask_i, as_tuple=False)
                mask_image[:, indices[:, 1], indices[:, 2]] = 0
                maska[step] = mask_image
                
                # Decode the current latent representation
                shape = (1, 16, 16, 256)
                z_q = self.model.vqgan.codebook.embedding(z_indices_predict).view(shape)
                z_q = z_q.permute(0, 3, 1, 2)
                decoded_img = self.model.vqgan.decode(z_q)
                dec_img_ori = (decoded_img[0] * std) + mean
                imga[step + 1] = dec_img_ori
                print(f"Step {step}: Mask sum after update: {mask_bc.sum().item()}")
             

                # Save the mask and decoded image for this iteration
                
                vutils.save_image(dec_img_ori, os.path.join("test_results", f"image_{i:03d}_step_{step:02d}.png"), nrow=1)

            # Save the final images
            vutils.save_image(maska, os.path.join("mask_scheduling", f"final_test_{i}.png"), nrow=10)
            vutils.save_image(imga, os.path.join("imga", f"final_test_{i}.png"), nrow=7)



class MaskedImage:
    def __init__(self, args):
        mi_ori=LoadTestData(root= args.test_maskedimage_path, partial=args.partial)
        self.mi_ori =  DataLoader(mi_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        mask_ori =LoadMaskData(root= args.test_mask_path, partial=args.partial)
        self.mask_ori =  DataLoader(mask_ori,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            drop_last=True,
                            pin_memory=True,
                            shuffle=False)
        self.device=args.device

    def get_mask_latent(self,mask):    
        downsampled1 = torch.nn.functional.avg_pool2d(mask, kernel_size=2, stride=2)
        resized_mask = torch.nn.functional.avg_pool2d(downsampled1, kernel_size=2, stride=2)
        resized_mask[resized_mask != 1] = 0       #1,3,16*16   check use  
        mask_tokens=(resized_mask[0][0]//1).flatten()   ##[256] =16*16 token
        mask_tokens=mask_tokens.unsqueeze(0)
        mask_b = torch.zeros(mask_tokens.shape, dtype=torch.bool, device=self.device)
        mask_b |= (mask_tokens == 0) #true means mask
        return mask_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT for Inpainting")
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on.')#cuda
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--num_workers', type=int, default=10, help='Number of worker')
    
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for MaskGIT')
    
    
#TODO3 step1-2: modify the path, MVTM parameters
    parser.add_argument('--load-transformer-ckpt-path', type=str, default='./transformer_checkpoints/best_model.pth', help='load ckpt')
    
    #dataset path
    parser.add_argument('--test-maskedimage-path', type=str, default='./lab5_dataset/cat_face/masked_image', help='Path to testing image dataset.')
    parser.add_argument('--test-mask-path', type=str, default='./lab5_dataset/mask64', help='Path to testing mask dataset.')
    #MVTM parameter
    parser.add_argument('--sweet-spot', type=int, default=10, help='sweet spot: the best step in total iteration')
    parser.add_argument('--total-iter', type=int, default=10, help='total step for mask scheduling')
    parser.add_argument('--mask-func', type=str, default='cosine', help='mask scheduling function')

    args = parser.parse_args()

    t=MaskedImage(args)
    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    maskgit = MaskGIT(args, MaskGit_CONFIGS)

    i=0
    for image, mask in zip(t.mi_ori, t.mask_ori):
        image=image.to(device=args.device)
        mask=mask.to(device=args.device)
        mask_b=t.get_mask_latent(mask)       
        maskgit.inpainting(image,mask_b,i)
        i+=1
        

# python inpainting 
# python inpainting 
# python inpainting 

# python faster-pytorch-fid/fid_score_gpu.py --predicted-path test_results --device cuda:0 --batch-size 1