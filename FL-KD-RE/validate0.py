import numpy as np
import torch
from timm.utils import *
from collections import OrderedDict
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