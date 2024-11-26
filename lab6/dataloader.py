import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SyntheticObjectsDataset(Dataset):
    def __init__(self, images_dir, labels_file, object_dict, transform=None):

        self.images_dir = images_dir
        self.transform = transform
        self.object_dict = object_dict
        
        # Load the label file
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # List of image file names
        self.image_files = list(self.labels.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Load the image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        # Get the labels and encode them
        label_names = self.labels[img_name]
        labels = torch.zeros(len(self.object_dict))
        
        for label in label_names:
            label_index = self.object_dict[label]
            labels[label_index] = 1.0
        
        return image, labels


class SyntheticObjectsLabelsOnlyDataset(Dataset):
    def __init__(self, labels_file, object_dict):

        self.object_dict = object_dict
        
        # Load the label file
        with open(labels_file, 'r') as f:
            self.labels = json.load(f)
        
        # Check if the labels structure is a list of lists
        if isinstance(self.labels, list) and isinstance(self.labels[0], list):
            self.label_lists = self.labels
        else:
            raise ValueError("Unsupported labels format in JSON file.")

    def __len__(self):
        return len(self.label_lists)

    def __getitem__(self, idx):

        label_names = self.label_lists[idx]
        
        # Encode labels as a multi-label tensor
        labels = torch.zeros(len(self.object_dict))
        for label in label_names:
            label_index = self.object_dict[label]
            labels[label_index] = 1.0
        
        return labels


def get_dataloader(images_dir=None, labels_file=None, batch_size=32, num_workers=4, return_labels_only=False):

    # Define the object dictionary
    object_dict = {
        "gray cube": 0, "red cube": 1, "blue cube": 2, "green cube": 3, 
        "brown cube": 4, "purple cube": 5, "cyan cube": 6, "yellow cube": 7, 
        "gray sphere": 8, "red sphere": 9, "blue sphere": 10, "green sphere": 11, 
        "brown sphere": 12, "purple sphere": 13, "cyan sphere": 14, "yellow sphere": 15, 
        "gray cylinder": 16, "red cylinder": 17, "blue cylinder": 18, "green cylinder": 19, 
        "brown cylinder": 20, "purple cylinder": 21, "cyan cylinder": 22, "yellow cylinder": 23
    }

    if not return_labels_only:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = SyntheticObjectsDataset(
            images_dir=images_dir,
            labels_file=labels_file,
            object_dict=object_dict,
            transform=transform
        )
    else:
        dataset = SyntheticObjectsLabelsOnlyDataset(
            labels_file=labels_file,
            object_dict=object_dict
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=not return_labels_only, num_workers=num_workers)
    
    return dataloader


# Example usage:
if __name__ == "__main__":
    images_dir = './iclevr'
    
    # Training data loader
    train_labels_file = './train.json'
    train_loader = get_dataloader(images_dir, train_labels_file, return_labels_only=False)

    # Test data loader (only labels)
    test_labels_file = './test.json'
    test_loader = get_dataloader(labels_file=test_labels_file, return_labels_only=True)

    # New test data loader (only labels)
    new_test_labels_file = './new_test.json'
    new_test_loader = get_dataloader(labels_file=new_test_labels_file, return_labels_only=True)

    # Example usage for training data
    print("Training Data:")
    for images, labels in train_loader:
        print(images.size(), labels.size())
        # Training loop here

    # Example usage for test data
    print("Test Data:")
    for labels in test_loader:
        print(labels.size())
        # Validation/testing loop here

    print("New Test Data:")
    for labels in new_test_loader:
        print(labels.size())
        # Validation/testing loop here
