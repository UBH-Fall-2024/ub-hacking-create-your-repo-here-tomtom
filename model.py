import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize to a 128x128
    transforms.Grayscale(),  # Convert to grayscale
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize
])


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)       # input, output, kernel, stride
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)

        self.fc1 = nn.Linear(256 * 6 * 6, 512)    # 256 * 6 * 6 -> 512
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 2)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)

        X = X.view(-1, 256 * 6 * 6)

        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        X = self.fc4(X)

        return F.log_softmax(X, dim=1)


model = CNN()
model.load_state_dict(torch.load("model_1.pt"))
model.eval()


def process_image(path):
    image = Image.open(path).convert('L')
    transform = transforms.Compose([                # Match transform composition
        transforms.Resize((128, 128)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = transform(image).unsqueeze(0)
    model.eval()

    with torch.no_grad():
        new_prediction = model(image)

    if new_prediction.argmax().item() == 0:
        return "Benign"
    else:
        return "Malignant"


image_path = "test_images/0000-b.jpg"
print(process_image(image_path))