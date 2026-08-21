import torch.nn as nn
import torch

class VGG(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: Conv2d (1 → 64) → ReLU → Conv2d (64 → 64) → ReLU → MaxPool 
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: Conv2d (64 → 128) → ReLU → Conv2d (128 → 128) → ReLU → MaxPool 
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: Conv2d (128 → 256) → ReLU → Conv2d (256 → 256) → ReLU → Conv2d (256 → 256) → ReLU → MaxPool 
            # Note the extra third convolution in this block.
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def load_vgg():
    model = VGG(num_classes=2)
    model.load_state_dict(torch.load('vgg.pth'))
    return model