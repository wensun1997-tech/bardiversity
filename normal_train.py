import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.utils.data as data

import extorch.vision.dataset as dataset
import extorch.utils as utils

from model.resnet import *


def train_epoch(net, trainloader, device, optimizer, criterion, epoch, report_every, logger, grad_clip = None):
    objs = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    
    net.train()
    for step, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = net(inputs)

        loss = criterion(logits, labels)
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


def valid(net, testloader, device, criterion, epoch, report_every, logger):
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
          
            loss = criterion(logits, labels)

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
    parser.add_argument("--load", default = None, type = str, 
            help = "load checkpoint from disk")
    parser.add_argument("--num-workers", default = 2, type = int)
    parser.add_argument("--save-every", type = int, default = 50)
    parser.add_argument("--report-every", default = 100, type = int)
    
    # hyper-parameter settings
    parser.add_argument("--epochs", default = 90, type = int)
    parser.add_argument("--batch-size", default = 64, type = int)
    parser.add_argument("--weight-decay", default = 5e-4, type = float)
    parser.add_argument("--grad-clip", default = 5., type = float)

    # learning scheduler settings
    parser.add_argument("--lr", default = 0.001, type = float, help = "initial learning rate")
    parser.add_argument("--milestones", type = int, nargs = "+", default = [80], 
            help = "milestones of the learning rate scheduler")
    parser.add_argument("--gamma", type = float, default = 0.1, 
            help = "gamma of the learning rate scheduler")
    args = parser.parse_args()

    LOGGER = utils.getLogger("Normal Train")

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
    if args.load:
        LOGGER.info("Load checkpoint from {}".format(args.load))
        net = torch.load(args.load)
    else:
        net = CIFARResNet18(num_classes = datasets.num_classes())
    net = net.to(DEVICE)

    num_params = utils.get_params(net)
    LOGGER.info("Parameter size: {:.5f}M".format(num_params / 1.e6))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(net.parameters()), lr = args.lr, weight_decay = args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones = args.milestones, gamma = args.gamma)
    
    time_estimator = utils.TimeEstimator(args.epochs)

    for epoch in range(1, args.epochs + 1):
        LOGGER.info("Epoch {} lr {:.5f}".format(epoch, optimizer.param_groups[0]["lr"]))

        loss, acc, acc_top5 = train_epoch(net, trainloader, DEVICE, optimizer, criterion, 
                epoch, args.report_every, LOGGER, args.grad_clip) 
        LOGGER.info("Train epoch {}: obj {:.3f}; Acc. Top-1 {:.3f}%; Top-5 {:.3f}%".format(
                epoch, loss, acc, acc_top5))

        loss, acc, acc_top5 = valid(net, testloader, DEVICE, criterion, epoch, args.report_every, LOGGER) 
        LOGGER.info("Test epoch {}: obj {:.3f}; Acc. Top-1 {:.3f}%; Top-5 {:.3f}%".format(
                epoch, loss, acc, acc_top5))

        if epoch % args.save_every == 0 and args.train_dir:
            save_path = os.path.join(args.train_dir, "{}.ckpt".format(epoch))
            torch.save(net, save_path)
            LOGGER.info("Save checkpoint at {}".format(save_path))
        
        LOGGER.info("Iter {} / {} Remaining time: {} / {}".format(epoch, args.epochs, *time_estimator.step()))

        scheduler.step()

    if args.train_dir:
        save_path = os.path.join(args.train_dir, "final.ckpt")
        torch.save(net, save_path)
        LOGGER.info("Save checkpoint at {}".format(save_path))


if __name__ == "__main__":
    main()
