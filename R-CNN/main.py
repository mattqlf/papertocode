import torch as tc
from torchvision.datasets import VOCDetection
from torchvision import transforms
import helpers
import debug
from model import CNN

pascal_voc2012 = VOCDetection(
    root="./data", download=False, transform=transforms.PILToTensor()
)

image = pascal_voc2012[0][0]

# regions = helpers.extract_regions(image, top_n=5)

# region = regions[0]

# model = CNN(classes=0)

# output = model(image.float())
# debug.matrix_verbose(output)

print(pascal_voc2012[0])
