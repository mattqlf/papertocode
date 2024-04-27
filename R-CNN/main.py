import torch as tc
from torchvision.datasets import VOCDetection, CIFAR10
from torch.utils.data import DataLoader, random_split
import torchvision
from torch import nn
from torchvision import transforms
from torchvision.models import alexnet
import torch.nn.functional as F
from helpers import get_class
from debug import show, matrix_verbose
from model import CNN, AlexFineTuned
from training import train_val
from metrics import accuracy
from tqdm import tqdm

def main():
    device = "cuda" if tc.cuda.is_available() else "cpu"

    pascal_voc2012 = VOCDetection(
        root="./data", download=False, transform=transforms.ToTensor()
    )

    model = alexnet(weights = 'DEFAULT')
    model = AlexFineTuned(classes = 20, model = model)
    # model = CNN(feature_dim=4096, classes=10)

    # loss_fn = tc.nn.CrossEntropyLoss()
    # metric = accuracy()
    # optimizer = tc.optim.SGD(model.parameters(), momentum=0.9, weight_decay=0.0005, lr = 0.01)
    # scheduler = tc.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7079) # go to 1e-5 in 20 epochs
    # scaler = tc.cuda.amp.GradScaler()
    
    # data = next(iter(train_loader))
    # output = model(data[0])
    # loss = loss_fn(output, data[1])
    # score = metric(output, data[1])
    # print(loss, score)

    # progress = train_val(model=model, train_loader=train_loader, val_loader=val_loader,loss_fn=loss_fn, metric=metric, optimizer=optimizer, scaler=scaler, scheduler= scheduler, epochs = 20, device=device)
    # print(progress)
    
    # alexnet_transforms = torchvision.models.AlexNet_Weights.DEFAULT.transforms()
    # data = pascal_voc2012[1234]
    # image = alexnet_transforms(data[0])
    # matrix_verbose(data[0])
    # print(get_class(data[1]))

    # output = model(image.unsqueeze(0))
    # prob = F.softmax(output)
    # index = tc.argmax(prob, dim=1)
    # matrix_verbose(prob)
    # print(index)

    print(model)
    
    
if __name__ == "__main__":
    main()
