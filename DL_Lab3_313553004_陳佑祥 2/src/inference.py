import argparse
import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from src.oxford_pet import load_dataset
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34UNet

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--model_type', type=str, choices=['unet', 'resnet34_unet'], required=True, help='model type to use')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    return parser.parse_args()

def visualize(images, masks, predictions):
    fig, axs = plt.subplots(10, 3, figsize=(15, 50))
    for i in range(10):
        image = images[i].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        mask = masks[i].squeeze()
        prediction = predictions[i].squeeze()
        
        axs[i, 0].imshow(image)
        axs[i, 0].set_title('Original Image')
        axs[i, 1].imshow(mask, cmap='gray')
        axs[i, 1].set_title('Ground Truth Mask')
        axs[i, 2].imshow(prediction, cmap='gray')
        axs[i, 2].set_title('Predicted Mask')
        for ax in axs[i, :]:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def infer(args):
    # Load dataset
    test_dataset = load_dataset(args.data_path, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    if args.model_type == 'unet':
        model = UNet(in_channels=3, out_channels=1)
    elif args.model_type == 'resnet34_unet':
        model = ResNet34UNet(in_channels=3, out_channels=1)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    images_to_display = []
    masks_to_display = []
    predictions_to_display = []

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5

            for i in range(images.size(0)):
                images_to_display.append(images[i].cpu().numpy())
                masks_to_display.append(masks[i].cpu().numpy())
                predictions_to_display.append(preds[i].cpu().numpy())
                
                if len(images_to_display) == 10:
                    visualize(images_to_display, masks_to_display, predictions_to_display)
                    return  # Exit after visualizing the first set of 10 images

if __name__ == '__main__':
    args = get_args()
    infer(args)


# python src/inference.py --model saved_models/DL_Lab3_UNet_313553004_陳佑祥.pth --model_type unet --data_path dataset/oxford-iiit-pet --batch_size 16
# python src/inference.py --model saved_models/DL_Lab3_ResNet34_UNet_313553004_陳佑祥.pth --model_type resnet34_unet --data_path dataset/oxford-iiit-pet --batch_size 16
