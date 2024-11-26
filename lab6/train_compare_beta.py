import os
import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Custom Imports
from dataloader import get_dataloader
from evaluator import evaluation_model
from model import CustomDDPM  # Adjust based on your actual model
from diffusers import DDPMScheduler

# Configuration
config = {
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.0002,
    "condition_dim": 4,
    "test_only": False,
    "model_checkpoint": 'ckpt',
    "log_dir": 'log/origin',
    "timesteps": 1000,
    "test_file": 'test.json',
    "test_batch_size": 32,
    "figure_save_dir": 'figure/origin',
    "resume_training": False,
    "checkpoint_file": 'net.pth'
}

def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config['model_checkpoint'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['figure_save_dir'], exist_ok=True)
    return device

def initialize_dataloader():
    train_loader = get_dataloader('./iclevr', './train.json', return_labels_only=False)
    test_loader = get_dataloader(labels_file=config['test_file'], return_labels_only=True)
    return train_loader, test_loader

def train_and_evaluate(scheduler_type):
    device = setup_environment()
    train_loader, test_loader = initialize_dataloader()
    model = CustomDDPM()
    
    # Initialize the scheduler with the specified type
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config['timesteps'], 
        beta_schedule=scheduler_type
    )

    results = []

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter(config['log_dir'])

    best_accuracy = 0.0

    for epoch in range(1, config['epochs'] + 1):
        model.train()
        epoch_loss = 0.0

        for idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
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
                print(f"    Batch {idx}, Loss: {loss.item():.5f}")
                writer.add_scalar(f'Training/{scheduler_type}_Loss', loss.item(), idx + epoch * len(train_loader))

        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar(f'Training/{scheduler_type}_Epoch_Loss', avg_loss, epoch)
        print(f"Epoch [{epoch}/{config['epochs']}] - Average Loss: {avg_loss:.5f}")

        accuracy = evaluate_model(model, test_loader, device, noise_scheduler, epoch)
        print(f"Epoch {epoch} - Test Accuracy: {accuracy:.2f}%")

        results.append({'epoch': epoch, 'scheduler': scheduler_type, 'loss': avg_loss, 'accuracy': accuracy})

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(config['model_checkpoint'], config['checkpoint_file']))
            print(f"Updated best model with accuracy: {best_accuracy:.2f}%")

    writer.close()
    return results

def evaluate_model(model, test_loader, device, noise_scheduler, epoch=None):
    model.eval()
    evaluator = evaluation_model()
    accuracy = 0.0
    filename = 'test' if config['test_file'] == 'test.json' else 'new_test'

    with torch.no_grad():
        for labels in test_loader:
            labels = labels.to(device)
            noise = torch.randn(labels.size(0), 3, 64, 64).to(device)

            step_image_dir = os.path.join(config['figure_save_dir'], f"{filename}_epoch_{epoch}")
            os.makedirs(step_image_dir, exist_ok=True)

            for step, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Denoising")):
                residual = model(noise, t, labels)
                noise = noise_scheduler.step(residual, t, noise).prev_sample

                if step == len(noise_scheduler.timesteps) - 1:
                    save_image(make_grid(noise, nrow=8, normalize=True),
                               os.path.join(step_image_dir, f"final_step_epoch_{epoch}.png"))

            images = noise
            accuracy = evaluator.eval(images, labels)

            if epoch is not None:
                save_image(make_grid(images, nrow=8, normalize=True),
                           os.path.join(config['figure_save_dir'], f'{filename}_{epoch}_{accuracy:.2f}.png'))
            else:
                save_image(make_grid(images, nrow=8, normalize=True),
                           os.path.join(config['figure_save_dir'], f'{filename}_{accuracy:.2f}_final.png'))

    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

def plot_results(results_df):
    plt.figure(figsize=(12, 6))

    for scheduler in results_df['scheduler'].unique():
        scheduler_data = results_df[results_df['scheduler'] == scheduler]
        plt.plot(scheduler_data['epoch'], scheduler_data['accuracy'], label=f'{scheduler} - Accuracy')
        plt.plot(scheduler_data['epoch'], scheduler_data['loss'], label=f'{scheduler} - Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Comparison of Different Schedulers')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config['figure_save_dir'], 'scheduler_comparison.png'))
    plt.show()

def main():
    # Train with different schedulers and collect the results
    results = []
    for scheduler_type in ['linear', 'scaled_linear', 'squaredcos_cap_v2']:
        print(f"Training with {scheduler_type} scheduler")
        results.extend(train_and_evaluate(scheduler_type))
    
    # Convert results to DataFrame and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(config['figure_save_dir'], 'scheduler_comparison.csv'), index=False)

    # Plot the results
    plot_results(results_df)

if __name__ == '__main__':
    main()
