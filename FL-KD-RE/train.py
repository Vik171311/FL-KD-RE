import numpy as np
import torch
from collections import OrderedDict
from timm.utils import *
torch.manual_seed(10)
random.random.seed(10)
torch.cuda.manual_seed(10)
torch.cuda.manual_seed_all(10)
np.random.seed(10)
def train_one_epoch(
        epoch, distiller, loader, optimizer, args,):

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    losses_m = AverageMeter()


    from collections import defaultdict
    losses_m_dict = defaultdict(AverageMeter)

    distiller.train()

    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    for batch_idx, (input, target, *additional_input) in enumerate(loader):
        last_batch = batch_idx == last_idx
        target = target.long()
        input, target = input.cuda(), target.cuda()
        additional_input = [i.cuda() for i in additional_input]

        output, losses_dict = distiller(input, target, *additional_input, epoch=epoch)
        loss = sum(losses_dict.values())

        losses_m.update(loss.item(), input.size(0))
        for k in losses_dict:
            losses_m_dict[k].update(losses_dict[k].item(), input.size(0))

        optimizer.zero_grad()

        loss.backward(create_graph=second_order)
        optimizer.step()


        num_updates += 1


    return OrderedDict([('loss', losses_m.avg)])