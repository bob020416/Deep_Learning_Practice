import torch
import torch.nn as nn

class SquareLayer(nn.Module):
    def __init__(self):
        super(SquareLayer, self).__init__()
    
    def forward(self, x):
        return torch.pow(x, 2)

class SCCNet(nn.Module):
    def __init__(self, numClasses, timeSample, Nu = 22, Nc = 22, Nt = 1, dropoutRate = 0.5):
        super(SCCNet, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, Nu, kernel_size=(Nc, Nt))
        self.batch_norm1 = nn.BatchNorm2d(Nu)

        # Second convolutional block
        self.conv2 = nn.Conv2d(22, 20, kernel_size=(1, 12), padding=(0, 6))
        self.batch_norm2 = nn.BatchNorm2d(20)
        self.square2 = SquareLayer()
        self.dropout2 = nn.Dropout(dropoutRate)

        # Pooling layer
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 62), stride=(1, 12))
 
        self.classifier = nn.Linear(640, numClasses, bias=True)


    def forward(self, x):
        x = x.unsqueeze(1)
        # First convolutional block
        x = self.conv1(x)
        x = self.batch_norm1(x)

        # Second convolutional block
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.square2(x)
        x = self.dropout2(x)

        # Pooling layer
        x = self.avg_pool(x)

        x = torch.log(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

if __name__ == "__main__":
    # Example usage:
    model = SCCNet(numClasses=4, timeSample=438, dropoutRate=0.5)
    print(model)

    # Create a dummy input with shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 1, 22, 438)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    print("Output:", output)
