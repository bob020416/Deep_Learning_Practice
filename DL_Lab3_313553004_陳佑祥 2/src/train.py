import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.oxford_pet import load_dataset
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34UNet
from src.utils import dice_score
from src.utils import CombinedLoss  # CombinedLoss

def train(args):
    # Load datasets
    print(f"Loading datasets from {args.data_path}...")
    train_dataset = load_dataset(args.data_path, mode="train")
    valid_dataset = load_dataset(args.data_path, mode="valid")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # Define model, loss function, and optimizer
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model == 'resnet34_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    criterion = CombinedLoss(weight_dice=1.0, weight_bce=1.0)  # Use CombinedLoss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Starting training for {args.epochs} epochs with model {args.model}...")

    # Initialize best_dice
    best_dice = 0.0

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Train one epoch
        model.train()
        running_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)  # Use CombinedLoss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # Print every batch (optional)
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        train_loss = running_loss / len(train_loader)
        print(f"  Training Loss: {train_loss:.4f}")

        # Validate the model
        model.eval()
        running_loss = 0.0
        total_dice_score = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)  # Use CombinedLoss
                running_loss += loss.item()

                preds = torch.sigmoid(outputs) > 0.5
                total_dice_score += dice_score(preds.cpu(), masks.cpu())
        valid_loss = running_loss / len(valid_loader)
        valid_dice = total_dice_score / len(valid_loader)
        print(f"  Validation Loss: {valid_loss:.4f}, Validation Dice: {valid_dice:.4f}")

        if valid_dice > best_dice:
            best_dice = valid_dice
            print(f"  New best model found! Saving to saved_models/DL_Lab3_{args.model}_best.pth")
            torch.save(model.state_dict(), f'saved_models/DL_Lab3_{args.model}_best.pth')

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, required=True, help='path of the input data')
    parser.add_argument('--model', type=str, choices=['unet', 'resnet34_unet'], required=True, help='model type to use')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=16, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='learning rate')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    train(args)

# python src/train.py --data_path dataset/oxford-iiit-pet --model unet --epochs 50 --batch_size 16 --learning_rate 0.0003325313446284999
# python src/train.py --data_path dataset/oxford-iiit-pet --model resnet34_unet --epochs 50 --batch_size 16 --learning_rate 0.00010623904863324497 