# # ------------------------- First use optuna fintuning best lr and train for another 200 epochs----------------------------
# import os
# import numpy as np
# from tqdm import tqdm
# import argparse
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import utils as torchvision_utils
# from models import MaskGit as VQGANTransformer
# from utils import LoadTrainData
# import yaml
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# import optuna

# import tensorboard

# class TrainTransformer:
#     def __init__(self, args, config):
#         self.config = config
#         self.args = args
#         self.model = self.initialize_model(config, args.device)
#         self.writer = SummaryWriter(log_dir="logs/")
        
#     def initialize_model(self, config, device):
#         model = VQGANTransformer(config["model_param"]).to(device=device)
#         return model

#     def initialize_optimizers(self, learning_rate):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
#         scheduler = None
#         return optimizer, scheduler

#     @staticmethod
#     def prepare_environment():
#         os.makedirs("transformer_checkpoints", exist_ok=True)

#     def run_training_loop(self, train_loader, val_loader, args):
#         def objective(trial):
#             # Suggest a learning rate
#             lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
#             self.optimizer, self.scheduler = self.initialize_optimizers(lr)
#             best_val_loss = np.inf
            
#             for epoch in range(1, 6):  # Train for 5 epochs during the study
#                 train_loss = self.run_epoch(train_loader, epoch, args, is_training=True)
#                 val_loss = self.run_epoch(val_loader, epoch, args, is_training=False)

#                 if val_loss < best_val_loss:
#                     best_val_loss = val_loss
#                     # Save best model during study
#                     self.save_checkpoint("best_val_optuna.pth")

#             return best_val_loss

#         # Run the Optuna study
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=10)
#         best_lr = study.best_params['learning_rate']
#         print(f"Best learning rate found by Optuna: {best_lr}")

#         # Train for an additional 200 epochs using the best learning rate found by Optuna
#         self.optimizer, self.scheduler = self.initialize_optimizers(best_lr)
#         for epoch in range(1, 201):
#             train_loss = self.run_epoch(train_loader, epoch, args, is_training=True)
#             val_loss = self.run_epoch(val_loader, epoch, args, is_training=False)

#             if val_loss < best_val_loss:
#                 best_val_loss = val_loss
#                 self.save_checkpoint("best_val_final.pth")

#     def run_epoch(self, data_loader, current_epoch, args, is_training=True):
#         if is_training:
#             self.model.train()
#         else:
#             self.model.eval()
        
#         epoch_losses = []
#         progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        
#         for batch_idx, batch_data in progress_bar:
#             batch_data = batch_data.to(args.device)
#             loss = self.compute_loss(batch_data)
#             epoch_losses.append(loss.item())

#             if is_training:
#                 self.perform_training_step(loss, batch_idx, args)
            
#             description = self.create_progress_description(current_epoch, batch_idx, len(data_loader), epoch_losses, is_training)
#             progress_bar.set_description_str(description)

#         self.log_loss(np.mean(epoch_losses), current_epoch, is_training)
#         return np.mean(epoch_losses)


#     def compute_loss(self, x):
#         logits, z_indices = self.model(x)
#         loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
#         return loss

#     def perform_training_step(self, loss, step, args):
#         loss.backward()
#         if step % args.accum_grad == 0:
#             self.optimizer.step()
#             self.optimizer.zero_grad()

#     def create_progress_description(self, epoch, step, total_steps, losses, train):
#         phase = "train" if train else "val"
#         description = f"{phase}_epoch: {epoch}, step: {step}/{total_steps}, loss: {np.mean(losses):.4f}"
#         return description

#     def log_loss(self, loss, epoch, train):
#         phase = "train" if train else "val"
#         self.writer.add_scalar(f"loss/{phase}", loss, epoch)

#     def save_checkpoint(self, filename):
#         torch.save(self.model.transformer.state_dict(), f"transformer_checkpoints/{filename}")


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="MaskGIT")
    
#     parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
#     parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
#     parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
#     parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
#     parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
#     parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
#     parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')
#     parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

#     parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
#     parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
#     parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
#     parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
#     parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate.')
#     parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

#     args = parser.parse_args()

#     MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
#     train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

#     train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
#     train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True, shuffle=True)
    
#     val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
#     val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True, shuffle=False)

#     train_transformer.run_training_loop(train_loader, val_loader, args)

#------------------------- Then train for another 200 epoch lr 0.000149456302123----------------------------
import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as torchvision_utils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import tensorboard
class TrainTransformer:
    def __init__(self, args, config):
        self.model = self.initialize_model(config, args.device)
        self.optimizer, self.scheduler = self.initialize_optimizers(args)
        self.prepare_environment()
        self.writer = SummaryWriter(log_dir="logs/")
        
    def initialize_model(self, config, device):
        model = VQGANTransformer(config["model_param"]).to(device=device)
        return model

    def initialize_optimizers(self, args):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        scheduler = None
        return optimizer, scheduler

    @staticmethod
    def prepare_environment():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def run_training_loop(self, train_loader, val_loader, args):
        best_train_loss = np.inf
        best_val_loss = np.inf
        for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
            train_loss = self.run_epoch(train_loader, epoch, args, is_training=True)
            val_loss = self.run_epoch(val_loader, epoch, args, is_training=False)
            
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                self.save_checkpoint("best_optuna.pth")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint("best_val.pth")

    def run_epoch(self, data_loader, current_epoch, args, is_training=True):
        if is_training:
            self.model.train()
        else:
            self.model.eval()
        
        epoch_losses = []
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        
        for batch_idx, batch_data in progress_bar:
            batch_data = batch_data.to(args.device)
            loss = self.compute_loss(batch_data)
            epoch_losses.append(loss.item())

            if is_training:
                self.perform_training_step(loss, batch_idx, args)
            
            description = self.create_progress_description(current_epoch, batch_idx, len(data_loader), epoch_losses, is_training)
            progress_bar.set_description_str(description)

        self.log_loss(np.mean(epoch_losses), current_epoch, is_training)
        return np.mean(epoch_losses)


    def compute_loss(self, x):
        logits, z_indices = self.model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), z_indices.view(-1))
        return loss

    def perform_training_step(self, loss, step, args):
        loss.backward()
        if step % args.accum_grad == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def create_progress_description(self, epoch, step, total_steps, losses, train):
        phase = "train" if train else "val"
        description = f"{phase}_epoch: {epoch}, step: {step}/{total_steps}, loss: {np.mean(losses):.4f}"
        return description

    def log_loss(self, loss, epoch, train):
        phase = "train" if train else "val"
        self.writer.add_scalar(f"loss/{phase}", loss, epoch)

    def save_checkpoint(self, filename):
        torch.save(self.model.transformer.state_dict(), f"transformer_checkpoints/{filename}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    
    parser.add_argument('--train_d_path', type=str, default="./lab5_dataset/cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./lab5_dataset/cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=10, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--save-per-epoch', type=int, default=1, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=0.000149456302123, help='Learning rate.')
    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True, shuffle=True)
    
    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True, shuffle=False)

    train_transformer.run_training_loop(train_loader, val_loader, args)
