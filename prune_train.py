import argparse
import os
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data as data

import extorch.vision.dataset as dataset
import extorch.utils as utils

from model.resnet import *
from criterion import BARStructuredLoss


def train_epoch(mask, net, teacher, trainloader, device, optimizer, criterion, epoch, report_every, logger, grad_clip = None):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    net.train()
    teacher.eval()

    for step, (inputs, labels) in enumerate(trainloader):
        current_epoch_fraction = epoch - 1 + step / len(trainloader)

        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = net(inputs)

        loss = criterion(mask, inputs, logits, labels, net, teacher, current_epoch_fraction)
        loss.backward()
        
        if grad_clip is not None:
            nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

        optimizer.step()
        prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
        n = inputs.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        del loss
        if (step + 1) % report_every == 0:
            logger.info("Epoch {} train {} / {} {:.3f}; {:.3f}%; {:.3f}%".format(
                epoch, step + 1, len(trainloader), objs.avg, top1.avg, top5.avg))
   
    return objs.avg, top1.avg, top5.avg


def valid(net, teacher, testloader, device, criterion, epoch, report_every, logger):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    net.eval()
    with torch.no_grad():
        for step, (inputs, labels) in enumerate(testloader):
            current_epoch_fraction = epoch - 1 + step / len(testloader)

            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = net(inputs)

            loss = criterion(inputs, logits, labels, net, teacher, current_epoch_fraction)

            prec1, prec5 = utils.accuracy(logits, labels, topk = (1, 5))
            n = inputs.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)
            del loss
            if (step + 1) % report_every == 0:
                logger.info("Epoch {} valid {} / {} {:.3f}; {:.3f}%; {:.3f}%".format(
                    epoch, step + 1, len(testloader), objs.avg, top1.avg, top5.avg))
    
    return objs.avg, top1.avg, top5.avg


def main():
    parser = argparse.ArgumentParser()
    
    # Basic settings
    parser.add_argument("--data-dir", type = str, required = True)
    parser.add_argument("--seed", default = None, type = int)
    parser.add_argument("--gpu", type = int, default = 0, help = "gpu device id")
    parser.add_argument("--train-dir", type = str, default = None, 
            help = "path to save the checkpoints")
    parser.add_argument("--load", required = True, type = str, 
            help = "load the pretrained network from disk")
    parser.add_argument("--load-net1", default='..........', type=str,
                        help="load the pretrained network from disk")
    parser.add_argument("--num-workers", default = 2, type = int)
    parser.add_argument("--save-every", type = int, default = 50)
    parser.add_argument("--report-every", default = 100, type = int)
    
    # pruning settings
    parser.add_argument("--budget", type = float, required = True, help = "target budget")
    parser.add_argument("--progress-func", type = str, choices = ["sigmoid", "exp"], 
            default = "sigmoid", help = "type of progress function")
    parser.add_argument("--_lambda", type = float, default = 1e-5, 
            help = "coefficient for trade-off of sparsity loss term")
    parser.add_argument("--distillation-temperature", type = float, default = 4., 
            help = "knowledge distillation temperature")
    parser.add_argument("--distillation-alpha", type = float, default = 0.9, 
            help = "knowledge distillation alpha")
    parser.add_argument("--tolerance", type = float, default = 0.01)
    parser.add_argument("--margin", type = float, default = 0.0001, 
            help = "parameter a in Eq. 5 of the paper")
    parser.add_argument("--sigmoid-a", type = float, default = 10., 
            help = "slope parameter of sigmoidal progress function")
    parser.add_argument("--upper-bound", type = float, default = 1e10,
            help = "upper bound for budget loss function")
    parser.add_argument("--alpha", type = float, default = 0., help = "alpha for GATE")
    parser.add_argument("--beta", type = float, default = 0.667, help = "beta for GATE")
    parser.add_argument("--gamma", type = float, default = -0.1, help = "gamma for GATE")
    parser.add_argument("--zeta", type = float, default = 1.1, help = "zeta for GATE")
    
    # hyper-parameter settings
    parser.add_argument("--epochs", default = 80, type = int)
    parser.add_argument("--batch-size", default = 64, type = int)
    parser.add_argument("--weight-decay", default = 5e-4, type = float)
    parser.add_argument("--grad-clip", default = 5., type = float)
    parser.add_argument("--lr", default = 0.001, type = float, help = "initial learning rate")
    args = parser.parse_args()

    LOGGER = utils.getLogger("Pruning Train")

    if args.train_dir:
        utils.makedir(args.train_dir, remove = True)
        LOGGER.addFile(os.path.join(args.train_dir, "train.log"))

    for name in vars(args):
        LOGGER.info("{}: {}".format(name, getattr(args, name)))

    DEVICE = torch.device("cuda:{}".format(args.gpu)) \
            if torch.cuda.is_available() else torch.device("cpu")

    if args.seed:
        utils.set_seed(args.seed)
        LOGGER.info("Set seed: {}".format(args.seed))

    datasets = dataset.CIFAR10(args.data_dir)
    trainloader = data.DataLoader(dataset = datasets.splits["train"], \
            batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True)
    testloader = data.DataLoader(dataset = datasets.splits["test"], \
            batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)

    # Construct the network
    net = torch.load(args.load)
    LOGGER.info("Load checkpoint from {}".format(args.load))
    
    teacher = copy.deepcopy(net)
    teacher.eval()
    teacher = teacher.to(DEVICE)

    net.add_wrapper(before_bn=False, alpha=args.alpha,
                    beta=args.beta, zeta=args.zeta, gamma=args.gamma)
    net = net.to(DEVICE)

    net1 = torch.load(args.load_net1)
    LOGGER.info("Load checkpoint from {}".format(args.load))
    mask1 = net1.hard_prune()
    mask = [mask1]

    criterion = BARStructuredLoss(
        args.budget, args.epochs, args.progress_func, args._lambda,
        args.distillation_temperature, args.distillation_alpha, args.tolerance,
        args.margin, args.sigmoid_a, args.upper_bound
    )
    
    normal_params = []
    gate_params = []
    for name, param in net.named_parameters():
        if 'log_alpha' in name:
            gate_params.append(param)
        else:
            normal_params.append(param)
    optimizer = optim.Adam([
        {"params": normal_params, "lr": args.lr, "weight_decay": args.weight_decay},
        {"params": gate_params, "lr": args.lr, "weight_decay": 0.}
    ])
    
    time_estimator = utils.TimeEstimator(args.epochs)

    for epoch in range(1, args.epochs + 1):
        LOGGER.info("Epoch {} lr {:.5f}".format(epoch, optimizer.param_groups[0]["lr"]))

        loss, acc, acc_top5 = train_epoch(mask, net, teacher, trainloader, DEVICE, optimizer,
                criterion, epoch, args.report_every, LOGGER, args.grad_clip) 
        LOGGER.info("Train epoch {}: obj {:.3f}; Acc. Top-1 {:.3f}%; Top-5 {:.3f}%".format(
                epoch, loss, acc, acc_top5))

        sparsity_ratio = criterion.sparsity_ratio(net)
        LOGGER.info("Epoch {}: sparsity ratio {:.3f}".format(epoch, sparsity_ratio))

        loss, acc, acc_top5 = valid(net, teacher, testloader, DEVICE, 
                criterion, epoch, args.report_every, LOGGER) 
        LOGGER.info("Test epoch {}: obj {:.3f}; Acc. Top-1 {:.3f}%; Top-5 {:.3f}%".format(
                epoch, loss, acc, acc_top5))

        if epoch % args.save_every == 0 and args.train_dir:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(epoch))
            torch.save(net, save_path)
            LOGGER.info("Save checkpoint at {}".format(save_path))
        
        LOGGER.info("Iter {} / {} Remaining time: {} / {}".format(epoch, args.epochs, *time_estimator.step()))

    if args.train_dir:
        save_path = os.path.join(args.train_dir, "final.ckpt")
        torch.save(net, save_path)
        LOGGER.info("Save checkpoint at {}".format(save_path))

        net.hard_prune()
        save_path = os.path.join(args.train_dir, "hard_pruned.ckpt")
        torch.save(net, save_path)
        LOGGER.info("Save checkpoint at {}".format(save_path))


if __name__ == "__main__":
    main()
