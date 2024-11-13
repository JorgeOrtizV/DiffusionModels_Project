import torch
from torchvision.utils import save_image
from torchvision import datasets, transforms

count = 1

test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)

for x, _ in test_dataset:
	save_image(x, 'original_testImgs/original_{}.png'.format(count))
	count+=1
