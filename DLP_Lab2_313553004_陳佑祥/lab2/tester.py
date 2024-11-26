import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.SCCNet import SCCNet
from Dataloader import MIBCI2aDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
    
    test_loss = running_loss / len(test_loader.dataset)
    accuracy = correct_predictions.double() / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return test_loss, accuracy

def load_test_data(training_method):
    if training_method == 'SD':
        test_features_dir = './dataset/SD_test/features/'
        test_labels_dir = './dataset/SD_test/labels/'
    elif training_method == 'LOSO':
        test_features_dir = './dataset/LOSO_test/features/'
        test_labels_dir = './dataset/LOSO_test/labels/'
    elif training_method == 'FT':
        test_features_dir = './dataset/LOSO_test/features/'
        test_labels_dir = './dataset/LOSO_test/labels/'
    else:
        raise ValueError('Invalid training method')

    test_dataset = MIBCI2aDataset(test_features_dir, test_labels_dir)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    return test_loader

def main():
    models_paths = {
        'FT': 'best_FT_model.pth',
        'SD': 'best_SD_model.pth',
        'LOSO': 'best_LOSO_model.pth'
    }
    
    num_classes = 4
    time_sample = 438
    dropout_rate = 0.5
    
    criterion = nn.CrossEntropyLoss()
    
    for method, model_path in models_paths.items():
        print(f'Evaluating model: {model_path}')
        
        # Load the model
        model = SCCNet(numClasses=num_classes, timeSample=time_sample, dropoutRate=dropout_rate).to(device)
        model.load_state_dict(torch.load(model_path))
        
        # Load the test data
        test_loader = load_test_data(method)
        
        # Evaluate the model
        test_loss, accuracy = evaluate(model, test_loader, criterion)
        print(f'Model: {method}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}\n')

if __name__ == '__main__':
    main()
