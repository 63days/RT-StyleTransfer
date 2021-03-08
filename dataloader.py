import torchvision
import torchvision.transforms as transforms
import torch
from PIL import Image
transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

def get_train_loader(batch_size):
    train_ds = torchvision.datasets.ImageFolder('dataset', transform=transform)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    style_img = Image.open('Camille_Mauclair.jpg')
    style_img = transform(style_img)
    return train_dl, style_img.unsqueeze(0)