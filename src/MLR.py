import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader


def accuracy(outputs, labels):
    _, preds = torch.max(
        outputs, dim=1
    )  # underscore discards the max value itself, we don't care about that
    return torch.sum(preds == labels).item() / len(preds)


def lossBatch(model, lossFn, xb, yb, opt=None, metric=None):
    # calculate the loss
    preds = model(xb)
    loss = lossFn(preds, yb)

    if opt is not None:
        # compute gradients
        loss.backward()
        # update parameters
        opt.step()
        # reset gradients to 0 (don't want to calculate second derivatives!)
        opt.zero_grad()

    metricResult = None
    if metric is not None:
        metricResult = metric(preds, yb)

    return loss.item(), len(xb), metricResult


def evaluate(model, lossFn, validDL, device, metric=None):
    # with torch.no_grad (this was causing an error)

    # pass each batch of the validation set through the model to form a multidimensional list (holding loss, length and metric for each batch)
    # the reason why we made optimization optional is so we can reuse the function here
    results = [
        lossBatch(model, lossFn, xb, yb.to(device), metric=metric,)
        for xb, yb in validDL
    ]

    # separate losses, counts and metrics
    losses, nums, metrics = zip(*results)

    # total size of the dataset (we keep track of lengths of batches since dataset might not be perfectly divisible by batch size)
    total = np.sum(nums)

    # find average total loss over all batches in validation (remember these are all vectors doing element wise operations.)
    avgLoss = np.sum(np.multiply(losses, nums)) / total

    # if there is a metric passed, compute the average metric
    if metric is not None:
        # avg of metric accross batches
        avgMetric = np.sum(np.multiply(metrics, nums)) / total

    return avgLoss, total, avgMetric


def fit(epochs, model, lossFn, opt, trainDL, valDL, device, metric=None):
    valList = [0.10]
    for epoch in range(epochs):
        # training - perform one step gradient descent on each batch, then moves on
        for xb, yb in trainDL:
            loss, _, lossMetric = lossBatch(model, lossFn, xb, yb.to(device), opt)

        # evaluation on cross val dataset - after updating over all batches, technically one epoch
        # evaluates over all validation batches and then calculates average val loss, as well as the metric (accuracy)
        valResult = evaluate(model, lossFn, valDL, device, metric)
        valLoss, total, valMetric = valResult
        valList.append(valMetric)
        # print progress
        if metric is None:
            print("Epoch [{}/{}], Loss: {:.4f}".format(epoch + 1, epochs, valLoss))
        else:
            print(
                "Epoch [{}/{}], Loss: {:.4f}, {}: {:.4f}".format(
                    epoch + 1, epochs, valLoss, metric.__name__, valMetric
                )
            )

    return valList


class CIFAR10(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        # this will dictate the rows of the theta matrix
        inputSize = 3 * 32 * 32

        # this will dictate the columns of the theta matrix
        numClasses = 10

        self.device = device
        self.linear = torch.nn.Linear(inputSize, numClasses, bias=False).to(device)

    def forward(self, xb):
        xb = xb.reshape(-1, 3072).to(self.device)
        out = self.linear(xb)
        return out


torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

batchSize = 100
trainLoader = DataLoader(trainset, batchSize)
testLoader = DataLoader(testset, batchSize)
model = CIFAR10(device)

for images, _ in trainLoader:
    outputs = model(images)
    break

# apply the softmax for each output row in our 100 x 10 output (with batch size 100)
probs = F.softmax(outputs, dim=1)

lrate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lrate)

lossFn = F.cross_entropy

trainList = fit(100, model, lossFn, optimizer, trainLoader, testLoader, device, metric=accuracy)
