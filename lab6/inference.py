# # ------------------------only one image-----------------------
# import os
# import torch
# from torchvision.utils import save_image, make_grid
# from tqdm import tqdm
# from PIL import Image

# # Custom Imports
# from dataloader import get_dataloader
# from evaluator import evaluation_model
# from model import CustomDDPM  # Adjust based on your actual model
# from diffusers import DDPMScheduler

# # Configuration
# config = {
#     "test_file": 'test.json',
#     "figure_save_dir": 'figure/origin',
#     "model_checkpoint": 'ckpt',
#     "checkpoint_file": 'net.pth',
#     "timesteps": 1000,
# }

# def setup_environment():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     os.makedirs(config['figure_save_dir'], exist_ok=True)
#     return device

# def initialize_dataloader():
#     test_loader = get_dataloader(labels_file=config['test_file'], return_labels_only=True)
#     return test_loader

# def evaluate_model_and_save_denoising_sequence(model, test_loader, device, noise_scheduler):
#     model.eval()
#     evaluator = evaluation_model()
#     accuracy = 0.0
#     filename = 'test_sequence'
#     images_sequence = []

#     with torch.no_grad():
#         for labels in test_loader:
#             labels = labels.to(device)
#             noise = torch.randn(1, 3, 64, 64).to(device)  # Use only one image (batch size 1)
#             label = labels[0].unsqueeze(0).to(device)  # Select the first label and reshape to maintain batch dimensions

#             for step, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Denoising")):
#                 residual = model(noise, t.to(device), label)  # Use the selected label
#                 noise = noise_scheduler.step(residual, t, noise).prev_sample

#                 # Save each intermediate step
#                 if step % 100 == 0 or step == len(noise_scheduler.timesteps) - 1:
#                     step_image = make_grid(noise, nrow=1, normalize=True)
#                     images_sequence.append(step_image)

#             # Save final image grid
#             final_image = make_grid(noise, nrow=1, normalize=True)
#             save_image(final_image, os.path.join(config['figure_save_dir'], f'{filename}_final.png'))

#             # Combine the images to create a sequence grid
#             sequence_image = torch.cat(images_sequence, dim=2)  # Combine horizontally
#             save_image(sequence_image, os.path.join(config['figure_save_dir'], f'{filename}_sequence.png'))

#             # Optionally, create and save a GIF of the sequence
#             pil_images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')) for img in images_sequence]
#             pil_images[0].save(os.path.join(config['figure_save_dir'], f'{filename}_sequence.gif'), save_all=True, append_images=pil_images[1:], duration=100, loop=0)

#             images = noise
#             accuracy = evaluator.eval(images, label)

#             break  # Only evaluate and save for the first image

#     print(f"Test Accuracy: {accuracy:.2f}%")
#     return accuracy

# def main():
#     device = setup_environment()

#     test_loader = initialize_dataloader()
#     model = CustomDDPM().to(device)  # Ensure the model is on the correct device
#     noise_scheduler = DDPMScheduler(num_train_timesteps=config['timesteps'])

#     model.load_state_dict(torch.load(os.path.join(config['model_checkpoint'], config['checkpoint_file']), map_location=device))
#     test_accuracy = evaluate_model_and_save_denoising_sequence(model, test_loader, device, noise_scheduler)
#     print(f"Test Accuracy: {test_accuracy:.2f}%")

# if __name__ == '__main__':
#     main()

# ---------------All 32 image Denoising ----------------------
import os
import torch
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
from PIL import Image

# Custom Imports
from dataloader import get_dataloader
from evaluator import evaluation_model
from model import CustomDDPM  # Adjust based on your actual model
from diffusers import DDPMScheduler

# seed 43 all 0.88
# seed 44 all 0.93
# seed 46 0.85 0.93
# Configuration
config = {
    "test_files": ['test.json', 'new_test.json'],  # Test both files
    "figure_save_dir": 'figure/origin',
    "model_checkpoint": 'ckpt',
    "checkpoint_file": 'DL_lab6_313553004_陳佑祥.pth',
    "timesteps": 1000,
    "seed": 46,  # Set a seed for reproducibility
}

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_environment():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(config['figure_save_dir'], exist_ok=True)
    return device

def initialize_dataloader(test_file):
    test_loader = get_dataloader(labels_file=test_file, return_labels_only=True)
    return test_loader

def evaluate_model_and_save_denoising_sequence(model, test_loader, device, noise_scheduler, filename_prefix):
    model.eval()
    evaluator = evaluation_model()
    accuracy = 0.0
    images_sequence = []

    with torch.no_grad():
        for labels in test_loader:
            labels = labels.to(device)
            noise = torch.randn(labels.size(0), 3, 64, 64).to(device)

            for step, t in enumerate(tqdm(noise_scheduler.timesteps, desc="Denoising")):
                residual = model(noise, t.to(device), labels)
                noise = noise_scheduler.step(residual, t, noise).prev_sample

                if step % 100 == 0 or step == len(noise_scheduler.timesteps) - 1:
                    step_image = make_grid(noise, nrow=8, normalize=True)
                    images_sequence.append(step_image)

            final_image = make_grid(noise, nrow=8, normalize=True)
            save_image(final_image, os.path.join(config['figure_save_dir'], f'{filename_prefix}_final.png'))

            sequence_image = torch.cat(images_sequence, dim=1)
            save_image(sequence_image, os.path.join(config['figure_save_dir'], f'{filename_prefix}_sequence.png'))

            pil_images = [Image.fromarray((img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')) for img in images_sequence]
            pil_images[0].save(os.path.join(config['figure_save_dir'], f'{filename_prefix}_sequence.gif'), save_all=True, append_images=pil_images[1:], duration=100, loop=0)

            images = noise
            accuracy = evaluator.eval(images, labels)

            break  # Only evaluate and save for the first batch

    print(f"{filename_prefix} Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    set_seed(config['seed'])  # Set the random seed for reproducibility

    device = setup_environment()
    model = CustomDDPM().to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=config['timesteps'])

    model.load_state_dict(torch.load(os.path.join(config['model_checkpoint'], config['checkpoint_file']), map_location=device))

    for test_file in config['test_files']:
        test_loader = initialize_dataloader(test_file)
        filename_prefix = os.path.splitext(os.path.basename(test_file))[0]
        test_accuracy = evaluate_model_and_save_denoising_sequence(model, test_loader, device, noise_scheduler, filename_prefix)
        print(f"{filename_prefix} Test Accuracy: {test_accuracy:.2f}%")

if __name__ == '__main__':
    main()
