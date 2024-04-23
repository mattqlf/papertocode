import torch as tc
from torchvision.datasets import VOCDetection
from torchvision import transforms
import helpers
import debug

pascal_voc2012 = VOCDetection(
    root="./data", download=False, transform=transforms.PILToTensor()
)

image = pascal_voc2012[0][0]

regions = helpers.extract_regions(image, top_n=50)

