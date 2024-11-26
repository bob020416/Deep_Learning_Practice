import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

from torch.utils.tensorboard import SummaryWriter
import tensorboard 

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing:
    def __init__(self, args, current_epoch=0):
        self.args = args
        self.current_epoch = current_epoch
        self.beta = 0.0
        self.update()

    def update(self):
        if self.args.annealing_strategy == 'cyclical':
            print("Using cyclical annealing strategy.")
            self.beta = self.frange_cycle_linear(
                n_iter=self.args.num_epoch, 
                start=0.1487, 
                stop=1.0, 
                n_cycle=self.args.n_cycles, 
                ratio=self.args.ratio
            )[self.current_epoch]
        elif self.args.annealing_strategy == 'monotonic':
            print("Using monotonic annealing strategy.")
            self.beta = min(1.0, self.current_epoch / self.args.num_epoch)
        elif self.args.annealing_strategy == 'without':
            print("No annealing strategy.")
            self.beta = 1.0  # Default to no annealing

        print(f"Updated Beta at epoch {self.current_epoch}: {self.beta}")

    def get_beta(self):
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0, n_cycle=1, ratio=1):
        L = np.ones(n_iter)
        period = n_iter / n_cycle
        step = (stop - start) / (period * ratio)  # linear schedule
        
        for c in range(n_cycle):
            v, i = start, 0
            while v <= stop and (int(i + c * period) < n_iter):
                L[int(i + c * period)] = v
                v += step
                i += 1
        return L


        

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        self.current_epoch = 0
        self.kl_annealing = kl_annealing(args, current_epoch=self.current_epoch)
        
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator            = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len
        self.val_vi_len   = args.val_vi_len
        self.batch_size = args.batch_size

        self.writer = SummaryWriter(log_dir=args.save_root)
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        for epoch in range(self.args.num_epoch):
            print(f"Starting epoch {epoch + 1}/{self.args.num_epoch}")
            self.current_epoch = epoch  # Ensure current_epoch is updated correctly
            self.kl_annealing.current_epoch = self.current_epoch
            self.kl_annealing.update()  # Ensure update is called after setting current_epoch
            
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False
            
            pbar = tqdm(train_loader, ncols=120)
            for step, (img, label) in enumerate(pbar):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing)
                
                beta = self.kl_annealing.get_beta()
                print(f"Epoch {self.current_epoch}, Step {step}, Beta: {beta}")
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss, lr=self.scheduler.get_last_lr()[0])
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, beta), pbar, loss, lr=self.scheduler.get_last_lr()[0])

                # Log training loss to TensorBoard
                self.writer.add_scalar('Loss/train', loss, epoch * len(train_loader) + step)
                self.writer.add_scalar('Beta', beta, epoch * len(train_loader) + step)
            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            
           

            self.kl_annealing.update()    
            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

        self.writer.close()

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()



                
      
    def training_one_step(self, img_batch, label_batch, adapt_TeacherForcing):
        def compute_losses(previous_frame, previous_label, time_steps, adapt_TeacherForcing, first_image):
            #  (2,16,3,32,64) input shape 
            mse_loss, kl_loss = 0.0, 0.0
            # in timestemp processing
            for t in range(1, time_steps):
                encoded_frame = self.frame_transformation(previous_frame[t].unsqueeze(0))
                encoded_label = self.label_transformation(previous_label[t].unsqueeze(0))
                z, mu, logvar = self.Gaussian_Predictor(encoded_frame, encoded_label)
                kl_loss += kl_criterion(mu, logvar,2)
                # if teacher forcing 
                previous = previous_frame[t-1].unsqueeze(0) if adapt_TeacherForcing else first_image
                previous_frame_encoding = self.frame_transformation(previous).detach()
                fusion_output = self.Decoder_Fusion(previous_frame_encoding, encoded_label, z)
                first_image = self.Generator(fusion_output)

                mse_loss += self.mse_criterion(first_image, previous_frame[t].unsqueeze(0))

            return mse_loss, kl_loss

        batch_size = img_batch.shape[0]
        time_steps = label_batch.size(1)
        total_mse_loss = 0.0
        total_kl_loss = 0.0
        beta = self.kl_annealing.get_beta()
        # in batch processing in timestamp
        for b in range(batch_size):
            previous_frame = img_batch[b]
            previous_label = label_batch[b]
            first_image = previous_frame[0].unsqueeze(0)

            mse_loss, kl_loss = compute_losses(previous_frame, previous_label, time_steps, adapt_TeacherForcing, first_image)

            total_mse_loss += mse_loss
            total_kl_loss += kl_loss

        average_mse_loss = total_mse_loss / batch_size
        average_kl_loss = total_kl_loss / batch_size
        total_loss = average_mse_loss + beta * average_kl_loss

        if torch.isnan(total_loss):
            print("NaN value encountered in loss. Skipping this batch.")
            return total_loss.item()

        self.optim.zero_grad()
        total_loss.backward()
        # add clipping or gradient will explode, loss nan ,this is very important!
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optim.step()

        return total_loss.item()



    @torch.no_grad()
    def val_one_step(self, img_batch, label_batch):
        def compute_val_losses(previous_frame, previous_label, time_steps, first_image):
            mse_loss, kl_loss, total_psnr = 0.0, 0.0, 0.0
            for t in range(1, time_steps):
                previous_image = first_image.detach()
                encoded_frame = self.frame_transformation(previous_frame[t].unsqueeze(0))
                encoded_label = self.label_transformation(previous_label[t].unsqueeze(0))
                z, mu, logvar = self.Gaussian_Predictor(encoded_frame, encoded_label)
                kl_loss += kl_criterion(mu, logvar,2)

                previous_frame_encoding = self.frame_transformation(previous_image).detach()
                fusion_output = self.Decoder_Fusion(previous_frame_encoding, encoded_label, z)
                first_image = self.Generator(fusion_output)

                mse_loss += self.mse_criterion(first_image, previous_frame[t].unsqueeze(0))
                psnr = Generate_PSNR(first_image, previous_frame[t].unsqueeze(0))
                total_psnr += psnr.item()

            return mse_loss, kl_loss, total_psnr

        batch_size = img_batch.shape[0]
        time_steps = label_batch.size(1)
        total_mse_loss, total_kl_loss, total_psnr = 0.0, 0.0, 0.0
        beta = self.kl_annealing.get_beta()

        for b in range(batch_size):
            previous_frame = img_batch[b]
            previous_label = label_batch[b]
            first_image = previous_frame[0].unsqueeze(0)

            mse_loss, kl_loss, batch_psnr = compute_val_losses(previous_frame, previous_label, time_steps, first_image)

            total_mse_loss += mse_loss
            total_kl_loss += kl_loss
            total_psnr += batch_psnr

        average_mse_loss = total_mse_loss / batch_size
        average_kl_loss = total_kl_loss / batch_size
        total_loss = average_mse_loss + beta * average_kl_loss
        average_psnr = total_psnr / (batch_size * (time_steps - 1))

        return float(total_loss), average_psnr

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        total_psnr = 0
        total_loss = 0
        num_batches = len(val_loader)

        pbar = tqdm(val_loader, ncols=120)
        for step, (img, label) in enumerate(pbar):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss, psnr = self.val_one_step(img, label)

            total_loss += loss
            total_psnr += psnr

            self.tqdm_bar('val', pbar, loss, lr=self.scheduler.get_last_lr()[0])

        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches

        # Log validation loss and PSNR to TensorBoard
        self.writer.add_scalar('Loss/val', avg_loss, self.current_epoch)
        self.writer.add_scalar('PSNR/val', avg_psnr, self.current_epoch)

        print(f"Validation Loss: {avg_loss}, Validation PSNR: {avg_psnr}")

                
    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        if self.current_epoch >= self.tfr_sde:
            new_tfr = self.tfr - self.tfr_d_step
            self.tfr = max(new_tfr, 0.0)

            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()



def main(args):
    
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.0001, help="initial learning rate")
    parser.add_argument('--device', type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim', type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--store_visualization', action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR', type=str, required=True, help="Your Dataset Path")
    parser.add_argument('--save_root', type=str, required=True, help="The path to save your data")
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--num_epoch', type=int, default=70, help="number of total epoch")
    parser.add_argument('--per_save', type=int, default=3, help="Save checkpoint every seted epoch")
    parser.add_argument('--partial', type=float, default=1.0, help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len', type=int, default=16, help="Training video length")
    parser.add_argument('--val_vi_len', type=int, default=630, help="valdation video length")
    parser.add_argument('--frame_H', type=int, default=32, help="Height input image to be resize")
    parser.add_argument('--frame_W', type=int, default=64, help="Width input image to be resize")
    
    # Module parameters setting
    parser.add_argument('--F_dim', type=int, default=128, help="Dimension of feature human frame")
    parser.add_argument('--L_dim', type=int, default=32, help="Dimension of feature label frame")
    parser.add_argument('--N_dim', type=int, default=12, help="Dimension of the Noise")
    parser.add_argument('--D_out_dim', type=int, default=192, help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    # Turn it off first
    parser.add_argument('--tfr', type=float, default=0, help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde', type=int, default=10, help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step', type=float, default=0.1, help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path', type=str, default=None, help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train', action='store_true')
    parser.add_argument('--fast_partial', type=float, default=0.4, help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch', type=int, default=5, help="Number of epoch to use fast train mode")
    
    # KL annealing strategy arguments
    parser.add_argument('--annealing_strategy', type=str, default='cyclical', help="KL annealing strategy")
    parser.add_argument('--n_cycles', type=int, default=10, help="Number of cycles for cyclical annealing")
    parser.add_argument('--ratio', type=float, default=1.0, help="Ratio for cyclical annealing")
    
    args = parser.parse_args()
    
    main(args)

# python Lab4_template/Trainer.py --DR LAB4_Dataset/LAB4_Dataset --save_root Lab4_template/check_point --device cuda --fast_train --annealing_strategy monotonic
# without beta = 1  tfr = 1 , loss 有在下降 但效果很差 psnr tune不上去 
# 如果不設tfr = 1 gradient會很快爆掉
# cyclical beta = 0-1  tfr = 1 loss 不會爆掉但效果很差
# monotonic beta = 0-1  tfr = 1 loss 不會爆掉但效果好一點

# monotonic 關tfr = 0  失敗
# cyclical 關tfr = 0  22
# without 關tfr = 0  31
# 訓練重點 skip loss if nan and cliipoed gradient and parameter 
# python Lab4_template/Trainer.py --DR LAB4_Dataset/LAB4_Dataset --save_root Lab4_template/check_point --device cuda --fast_train --annealing_strategy without --num_epoch 400