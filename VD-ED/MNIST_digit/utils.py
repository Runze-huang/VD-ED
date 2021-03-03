import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader

BATCH_SIZE = 128
NUM_WORKERS = 2


def prepare_MNIST(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
	transform = transforms.Compose([       
		transforms.ToTensor(),
	])

	training_set = torchvision.datasets.MNIST('./dataset/', train=True, download=False, transform=transform)
	test_set = torchvision.datasets.MNIST('./dataset/', train=False, download=False, transform=transform)
	trainloader = DataLoader(training_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

	clssses = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
	return trainloader, testloader, clssses



if __name__ == '__main__':
	trainloader, testloader, classes = prepare_MNIST()
	data_iter = iter(trainloader)
	imgs, labels = data_iter.next()
	imshow(torchvision.utils.make_grid(imgs, nrow=16))
	print(" ".join("%5s" % classes[labels[j]] for j in range(4)))
