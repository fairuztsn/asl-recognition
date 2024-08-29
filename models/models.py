import torch

class SignLanguageClassifierV0(torch.nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()
        self.layer_stack = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=384, out_features=120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, num_classes),
            torch.nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        return self.layer_stack(x)