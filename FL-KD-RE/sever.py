
import random
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CIFAR100
from getsim import *
from model import *
from skd import *
from train import *
from validate1 import *  #有毒数据
# from validate0 import *  #无毒数据

import datetime
now = datetime.datetime.now()
time = now.strftime("%Y%m%d-%H%M%S")
filename = f"accuracy_{time}.txt"
txt=open(filename,'w+')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def federated_average(train_dataset, test_dataset, args,net_dataidx_map):
    pre_models={}
    model_name={}
    # student = {0:[0,1],1:[2,3],2:[4,5],3:[6,7]}
    # teacher = {0:[0,1,2],1:[3,4,5,6],2:[7,8,9,10,11,12],3:[13,14,15]}
    pre_models[0]=ResNet18().to(device)
    model_name[0]='resnet18'
    pre_models[1]=ResNet34().to(device)
    model_name[1]='resnet34'
    pre_models[2]=ResNet34_2().to(device)
    model_name[2]='resnet34_2'
    pre_models[3]=ResNet34_3().to(device)
    model_name[3]='resnet34_3'
    pre_models[4]=ResNet34_4().to(device)
    model_name[4]='resnet34_4'

    pre_models[1].load_state_dict(torch.load('clean_pretrain_c10_34.pth'))
    pre_models[2].load_state_dict(torch.load('backdoor_pretrain_c10_34.pth'))

    pre_models[3].load_state_dict(torch.load('clean_pretrain_c100_34.pth'))
    pre_models[4].load_state_dict(torch.load('backdoor_pretrain_c100_34.pth'))


    pre_models[1].requires_grad_(False)
    pre_models[1].eval()
    pre_models[2].requires_grad_(False)
    pre_models[2].eval()

    pre_models[3].requires_grad_(False)
    pre_models[3].eval()
    pre_models[4].requires_grad_(False)
    pre_models[4].eval()


    client_models={}
    client_modelsnames={}

    for i in range(args.clientnum):
        client_model = {}
        client_modelsname = {}
        net=ResNet18()
        net.to(device)

        client_model[0]=net
        client_model[1] = pre_models[1]
        client_model[2] = pre_models[2]
        client_model[3] = pre_models[3]
        client_model[4] = pre_models[4]
        client_model[1].requires_grad_(False)
        client_model[1].eval()
        client_model[2].requires_grad_(False)
        client_model[2].eval()

        client_model[3].requires_grad_(False)
        client_model[3].eval()
        client_model[4].requires_grad_(False)
        client_model[4].eval()
        client_models[i] = client_model

        client_modelsname[0] = model_name[0]
        client_modelsname[1] = model_name[1]
        client_modelsname[2] = model_name[2]
        client_modelsname[3] = model_name[3]
        client_modelsname[4] = model_name[4]
        client_modelsnames[i] = client_modelsname




    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    client_split_dict ={}
    client_optimizers={}
    client_schedulers={}
    client_loss = {}
    client_distillers={}
    for client_idx in range(args.clientnum):
        client_data_indices = net_dataidx_map[client_idx]
        client_dataset = Subset(train_dataset, client_data_indices)
        client_train_loader = DataLoader(client_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
        best_all_assignmemt, best_block_split_dict = compute_sim(args, client_models[client_idx], client_modelsnames[client_idx], client_train_loader)
        client_split_dict[client_idx]=best_block_split_dict

        train_loss_fn = nn.CrossEntropyLoss().cuda()
        client_loss[client_idx] = train_loss_fn
        distiller = SKD(client_models[client_idx][0], client_models[client_idx][1], client_models[client_idx][2],client_models[client_idx][3], client_models[client_idx][4], client_loss[client_idx], args,client_split_dict[client_idx])
        client_distillers[client_idx] = distiller
        optimizer = optim.SGD(distiller.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,nesterov=True)
        client_optimizers[client_idx] = optimizer
        scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        client_schedulers[client_idx] = scheduler

    for epoch in range(args.comm_round):
        print("第"+str(epoch+1)+"轮")
        txt.write("第" + str(epoch + 1) + "轮" + '\n')
        avgacc = []
        for i in range(args.clientnum):
            client_models[i][0].load_state_dict(pre_models[0].state_dict())
        for client_idx in range(args.clientnum):
            client_data_indices = net_dataidx_map[client_idx]
            client_dataset = Subset(train_dataset, client_data_indices)
            client_train_loader = DataLoader(client_dataset, batch_size=128, shuffle=True, num_workers=2)



            for epoch in range(args.epochs):
                # print("这是第" +str(client_idx)+'个客户端的第'+ str(epoch) + "轮")
                train_metrics = train_one_epoch(epoch, client_distillers[client_idx], client_train_loader, client_optimizers[client_idx], args)

            val_metrics = validate(client_models[client_idx][0], test_loader, client_loss[client_idx])
            print('第'+str(client_idx)+'个客户端的'+'准确率是'+str(val_metrics))
            # txt.write('第' + str(client_idx) + '个客户端的' + '准确率是' + str(val_metrics) + '\n')
            avgacc.append(val_metrics['top1'])
            test_loss, accuracy = test_model_intrigger(client_models[client_idx][0], test_loader)
            print('第'+str(client_idx)+'个客户端的'+'投毒准确率是'+str(accuracy))
            # txt.write('第' + str(client_idx) + '个客户端的' + '投毒准确率是' + str(accuracy) + '\n')

            client_schedulers[client_idx].step()
        avgacc = sum(avgacc) / args.clientnum
        print('>> 平均准确率: %f' % avgacc)
        txt.write('平均准确率是' + str(avgacc) + '\n')

        server_state_dict = pre_models[0].state_dict()
        total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.clientnum)])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.clientnum)]
        for key in server_state_dict:
            server_state_dict[key] = torch.stack(
                [client_models[i][0].state_dict()[key] * fed_avg_freqs[i] for i in range(args.clientnum)], 0).sum(0)
        pre_models[0].load_state_dict(server_state_dict)
        validate_loss_fn = nn.CrossEntropyLoss().cuda()
        val_metrics = validate(pre_models[0], test_loader, validate_loss_fn)

        print('服务器聚合模型的准确率是:' + str(val_metrics))
        txt.write('服务器聚合模型的准确率是:' + str(val_metrics) + '\n')
        test_loss, accuracy = test_model_intrigger(pre_models[0], test_loader)
        print('服务器聚合模型的投毒准确率是:' + str(accuracy))
        txt.write('服务器聚合模型的投毒准确率是:' + str(accuracy) + '\n')
