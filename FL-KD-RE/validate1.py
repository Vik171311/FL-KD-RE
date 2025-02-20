import numpy as np
import torch
from timm.utils import *
from collections import OrderedDict

from torch import nn

torch.manual_seed(10)
random.random.seed(10)
torch.cuda.manual_seed(10)
torch.cuda.manual_seed_all(10)
np.random.seed(10)
def validate(model, loader, loss_fn):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()

    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]


            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))


            reduced_loss = loss.data


            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))



    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


def test_model_intrigger(model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        for data, target in test_loader:
            for i in range(len(data)):
                data[i, 0, 0, 0] = 1
                data[i, 0, 0, 1] = 1
                data[i, 0, 1, 0] = 1
                data[i, 0, 1, 1] = 1
                data[i, 1, 0, 0] = 1
                data[i, 1, 0, 1] = 1
                data[i, 1, 1, 0] = 1
                data[i, 1, 1, 1] = 1
                data[i, 2, 0, 0] = 1
                data[i, 2, 0, 1] = 1
                data[i, 2, 1, 0] = 1
                data[i, 2, 1, 1] = 1
                target[i] = 1
            data, target = data.cuda(), target.cuda()
            target = target.long()
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    return test_loss, accuracy