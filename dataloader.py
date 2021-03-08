import torchvision
import torchvision.transforms as transforms
import torch

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor()
])

def get_train_loader(batch_size):
    train_ds = torchvision.datasets.ImageFolder('dataset', transform=transform)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    return train_dl