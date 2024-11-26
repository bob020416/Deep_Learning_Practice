import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from diffusers import DDPMScheduler, UNet2DModel

# Custom Imports
from dataloader import get_dataloader
from evaluator import evaluation_model

# Configuration
config = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.0002,
    "condition_dim": 4,
    "test_only": False,
    "model_checkpoint": 'ckpt',
    "log_dir": 'log',
    "timesteps": 1000,
    "test_file": 'test.json',
    "test_batch_size": 32,
    "figure_save_dir": 'figure',
    "resume_training": False,
    "checkpoint_file": 'net.pth',
    "csv_save_dir": 'results'
}

class CustomDDPM(nn.Module):
    def __init__(self, num_classes=24, model_dim=512, attention_levels=3):
        super(CustomDDPM, self).__init__()
        
        self.embedding_dim = model_dim // 4
        self.embedding_layer = nn.Linear(num_classes, model_dim)
        
        # Adjust attention levels based on the model configuration
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
                "DownBlock2D",  # First three blocks without attention
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D" if attention_levels >= 1 else "DownBlock2D",  # Last three blocks with increasing attention
                "AttnDownBlock2D" if attention_levels >= 2 else "DownBlock2D",
                "AttnDownBlock2D" if attention_levels >= 3 else "DownBlock2D"
            ],
            up_block_types=[
                "AttnUpBlock2D" if attention_levels >= 1 else "UpBlock2D",  # Last three blocks with increasing attention
                "AttnUpBlock2D" if attention_levels >= 2 else "UpBlock2D",
                "AttnUpBlock2D" if attention_levels >= 3 else "UpBlock2D",
                "UpBlock2D",  # First three blocks without attention
                "UpBlock2D",
                "UpBlock2D"
            ],
            class_embed_type="identity"
        )
    
    def forward(self, input_tensor, time_step, labels):
        embedded_class = self.embedding_layer(labels)
        output = self.unet_model(input_tensor, time_step, embedded_class)
        return output.sample

def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config['model_checkpoint'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['figure_save_dir'], exist_ok=True)
    os.makedirs(config['csv_save_dir'], exist_ok=True)
    return device

def initialize_dataloader():
    train_loader = get_dataloader('./iclevr', './train.json', return_labels_only=False)
    test_loader = get_dataloader(labels_file=config['test_file'], return_labels_only=True)
    return train_loader, test_loader

def train_and_evaluate_model(model, train_loader, test_loader, device, noise_scheduler, model_name):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter(os.path.join(config['log_dir'], model_name))
    
    best_accuracy = 0.0
    results = []

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0

        for idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"{model_name} - Training Epoch {epoch}")):
            images, labels = images.to(device), labels.to(device)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, config['timesteps'], (images.size(0),)).long().to(device)
            noisy_images = noise_scheduler.add_noise(images, noise, timesteps)

            optimizer.zero_grad()
            predictions = model(noisy_images, timesteps, labels)
            loss = criterion(predictions, noise)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if idx % 1000 == 0:
                print(f"{model_name} - Batch {idx}, Loss: {loss.item():.5f}")
                writer.add_scalar(f'Training/{model_name}_Loss', loss.item(), idx + epoch * len(train_loader))

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar(f'Training/{model_name}_Epoch_Loss', avg_loss, epoch)
        print(f"{model_name} - Epoch [{epoch}/{config['epochs']}] - Average Loss: {avg_loss:.5f}")

        accuracy = evaluate_model(model, test_loader, device, noise_scheduler, epoch)
        print(f"{model_name} - Epoch {epoch} - Test Accuracy: {accuracy:.2f}%")

        results.append({'epoch': epoch, 'loss': avg_loss, 'accuracy': accuracy})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(config['model_checkpoint'], f'{model_name}_{config["checkpoint_file"]}'))
            print(f"{model_name} - Updated best model with accuracy: {best_accuracy:.2f}%")

    writer.close()

    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_file_path = os.path.join(config['csv_save_dir'], f'{model_name}_results.csv')
    results_df.to_csv(csv_file_path, index=False)

    # Plot results
    plot_results(results_df, model_name)

def evaluate_model(model, test_loader, device, noise_scheduler, epoch=None):
    model.eval()
    evaluator = evaluation_model()
    accuracy = 0.0

    with torch.no_grad():
        for labels in test_loader:
            labels = labels.to(device)
            noise = torch.randn(labels.size(0), 3, 64, 64).to(device)

            for step, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Denoising")):
                residual = model(noise, t, labels)
                noise = noise_scheduler.step(residual, t, noise).prev_sample

            images = noise
            accuracy = evaluator.eval(images, labels)
            break

    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def plot_results(results_df, model_name):
    plt.figure(figsize=(10, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(results_df['epoch'], results_df['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} - Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(results_df['epoch'], results_df['accuracy'], label='Accuracy', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()

    # Save Plot
    plot_file_path = os.path.join(config['figure_save_dir'], f'{model_name}_training_plot.png')
    plt.savefig(plot_file_path)
    plt.show()

def main():
    device = setup_environment()
    train_loader, test_loader = initialize_dataloader()
    noise_scheduler = DDPMScheduler(num_train_timesteps=config['timesteps'])

    models = {
        'attention_last_3': CustomDDPM(attention_levels=3),
        'attention_last_2': CustomDDPM(attention_levels=2),
        'attention_last_1': CustomDDPM(attention_levels=1),
        'no_attention': CustomDDPM(attention_levels=0),
    }

    for model_name, model in models.items():
        print(f"Training {model_name} model")
        train_and_evaluate_model(model, train_loader, test_loader, device, noise_scheduler, model_name)

if __name__ == '__main__':
    main()
