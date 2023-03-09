import argparse
import os
import logging
from utils.tools import *
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from models_new import PointCAT
from utils.dataset import ModelNet40
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import sklearn.metrics as metrics
import numpy as np
from math import cos, pi


def adjust_learning_rate(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=False):
    warmup_epoch = 5 if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * (current_epoch + 1) / warmup_epoch
    else:
        lr = lr_min + (lr_max - lr_min)*(1 + cos(pi * (current_epoch -
                                                       warmup_epoch + 1) / (max_epoch - warmup_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def printf(str):
    screen_logger = logging.getLogger("Model")
    screen_logger.info(str)
    print(str)


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train(net, trainloader, optimizer, criterion, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        # so, the input data shape is [batch, 3, 1024]
        data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        logits = net(data)
        loss = criterion(logits, label)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():

        for batch_idx, (data, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--checkpoint', default=None, type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size in training')
    parser.add_argument('--model', default='Vertax_elite', help='model_name')
    parser.add_argument('--epoch', default=250, type=int,
                        help='number of epoch in training')
    parser.add_argument('--num_points', type=int,
                        default=1024, help='Point Number')
    parser.add_argument('--use_sgd', type=bool,
                        default=True, help='use sgd / adam')
    parser.add_argument('--learning_rate', default=0.01,
                        type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float,
                        default=2e-4, help='decay rate')
    parser.add_argument('--min_lr', default=1e-5, type=float, help='min lr')
    parser.add_argument('--new_lr', default=False,
                        type=bool, help='use new lr')
    parser.add_argument('--no_cuda', type=bool,
                        default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default='3407', help='random seed')
    parser.add_argument('--dropout', type=float,
                        default=0.5, help='dropout rate')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    parser.add_argument('--new_model', default=True,
                        type=bool, help='choose new model')
    args = parser.parse_args()

    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        print(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        device = 'cuda'
    else:
        printf('Using CPU')
        device = 'cpu'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    args.model = "Vertax_elite" if args.new_model else "Vertax"
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    if args.checkpoint is None:
        args.checkpoint = 'checkpoints/' + \
            args.model + message + '-' + str(args.seed)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(
        os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)
    printf(f"args: {args}")
    printf('==> Building model..')

    net = PointCAT().to(device)
    net.apply(weight_init)
    criterion = cal_loss  # cross_entropy

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'),
                        title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        state_dict = checkpoint['net']

        net.load_state_dict(state_dict, strict=False)
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'),
                        title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.num_points), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    if args.use_sgd:
        print('Use SGD')
        optimizer = torch.optim.SGD([{'params': net.parameters(), 'initial_lr': args.learning_rate}],
                                    lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        print('Use Adam')
        optimizer = torch.optim.Adam([{'params': net.parameters(), 'initial_lr': args.learning_rate}],
                                     lr=args.learning_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr)

    for epoch in range(start_epoch, args.epoch):
        # adjust_learning_rate(optimizer=optimizer, current_epoch=epoch, max_epoch=args.epoch, lr_min=args.min_lr, lr_max=args.learning_rate, warmup=True)
        printf('Epoch(%d/%s) Learning Rate %s:' %
               (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        # {"loss", "acc", "acc_avg", "time"}
        train_out = train(net, train_loader, optimizer, criterion, device)
        test_out = validate(net, test_loader, criterion, device)
        scheduler.step()
        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (
            test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (
            train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (
            test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (
            train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (
            test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (
            train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  # best test accuracy
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++++++" * 2)
    printf(
        f"++  Last Train time: {train_out['time']}s | Last Test time: {test_out['time']}s  ++")
    printf(
        f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(
        f"++  Best Train Acc_avg: {best_train_acc_avg}% | Best Test Acc_avg: {best_test_acc_avg}%  ++")
    printf(
        f"++  Best Train Acc: {best_train_acc}% | Best Test Acc: {best_test_acc}%  ++")
    printf(f"+++++++++" * 5)
