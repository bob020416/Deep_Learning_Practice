# evaluate.py
import torch
from torch.utils.data import DataLoader
from src.oxford_pet import load_dataset
from src.utils import dice_score
from src.models.unet import UNet
from src.models.resnet34_unet import ResNet34UNet

def evaluate(model, data_loader, device):
    model.eval()
    total_dice_score = 0.0
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs) > 0.5
            total_dice_score += dice_score(preds.cpu(), masks.cpu())
    return total_dice_score / len(data_loader)

def main(args):
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

    dice = evaluate(model, test_loader, device)
    print(f"Test Dice Score: {dice:.4f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate the model on the test dataset')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weight')
    parser.add_argument('--model_type', type=str, choices=['unet', 'resnet34_unet'], required=True, help='model type to use')
    parser.add_argument('--data_path', type=str, required=True, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    args = parser.parse_args()
    main(args)

# python src/evaluate.py --model saved_models/DL_Lab3_UNet_313553004_陳佑祥.pth --model_type unet --data_path dataset/oxford-iiit-pet --batch_size 16
# Unet  Test Dice Score: 0.9318
# python src/evaluate.py --model saved_models/DL_Lab3_ResNet34_UNet_313553004_陳佑祥.pth --model_type resnet34_unet --data_path dataset/oxford-iiit-pet --batch_size 16
# Res34Unet Test Dice Score: 0.9314