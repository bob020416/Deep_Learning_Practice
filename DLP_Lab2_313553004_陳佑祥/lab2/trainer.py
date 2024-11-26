import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.SCCNet import SCCNet
from Dataloader import MIBCI2aDataset
from utils import plot_loss, plot_accuracy, calculate_confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, criterion, optimizer, scheduler, num_epochs=25, patience=65):
    model.train()
    train_losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = model.state_dict()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        scheduler.step(epoch_loss)
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print("Early stopping")
            break
    
    model.load_state_dict(best_model_wts)
    print('Training complete')
    return model, train_losses


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

from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

def main():
    # Set parameters after optuna finetuning 
    # SD best: learning_rate: 0.007811190251140579 , dropout_rate: 0.4366490162959512
    # SOLO best: learning_rate: 0.0016697658251998556 dropout_rate: 0.4254228126351459
    num_classes = 4
    time_sample = 438
    num_epochs = 50
    batch_size = 64
    learning_rate = 0.0016697658251998556
    dropout_rate = 0.4254228126351459
    weight_decay = 1e-1
    warmup_epochs = 10
    
    training_method = 'FT'  # Change as needed
    
    if training_method == 'SD':
        train_features_dir = './dataset/SD_train/features/'
        train_labels_dir = './dataset/SD_train/labels/'
        test_features_dir = './dataset/SD_test/features/'
        test_labels_dir = './dataset/SD_test/labels/'
    elif training_method == 'LOSO':
        train_features_dir = './dataset/LOSO_train/features/'
        train_labels_dir = './dataset/LOSO_train/labels/'
        test_features_dir = './dataset/LOSO_test/features/'
        test_labels_dir = './dataset/LOSO_test/labels/'
    elif training_method == 'FT':
        loso_train_features_dir = './dataset/LOSO_train/features/'
        loso_train_labels_dir = './dataset/LOSO_train/labels/'
        finetune_features_dir = './dataset/FT/features/'
        finetune_labels_dir = './dataset/FT/labels/'
        test_features_dir = './dataset/LOSO_test/features/'
        test_labels_dir = './dataset/LOSO_test/labels/'
    else:
        raise ValueError('Invalid training method')
    
    # Load dataset
    if training_method == 'FT':
        loso_train_dataset = MIBCI2aDataset(loso_train_features_dir, loso_train_labels_dir)
        finetune_train_dataset = MIBCI2aDataset(finetune_features_dir, finetune_labels_dir)
        test_dataset = MIBCI2aDataset(test_features_dir, test_labels_dir)
        loso_train_loader = DataLoader(loso_train_dataset, batch_size=batch_size, shuffle=True)
        finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = MIBCI2aDataset(train_features_dir, train_labels_dir)
        test_dataset = MIBCI2aDataset(test_features_dir, test_labels_dir)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    model = SCCNet(numClasses=num_classes, timeSample=time_sample, dropoutRate=dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate warm-up
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.

    scheduler = LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    
    # Train and evaluate model
    if training_method == 'FT':
        model, train_losses = train(model, loso_train_loader, criterion, optimizer, scheduler, 300)
        model, finetune_losses = train(model, finetune_train_loader, criterion, optimizer, cosine_scheduler, 200)
        train_losses.extend(finetune_losses)
    else:
        model, train_losses = train(model, train_loader, criterion, optimizer, scheduler, num_epochs)
    
    test_loss, accuracy = evaluate(model, test_loader, criterion)
    
    # Save the trained model
    model_save_path = f'sccnet_{training_method}.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Plot training loss
    plot_loss(train_losses, [], title='Training Loss over Epochs')
    
    # Calculate and plot confusion matrix
    class_names = ['Left Hand', 'Right Hand', 'Feet', 'Tongue']
    calculate_confusion_matrix(model, test_loader, class_names)

if __name__ == '__main__':
    main()
