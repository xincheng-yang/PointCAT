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


def printf(str):
    screen_logger = logging.getLogger("Model")
    screen_logger.info(str)
    print(str)


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
    parser.add_argument('--model', default='PointCAT', help='model_name')
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
    parser.add_argument('--no_cuda', type=bool,
                        default=False, help='enables CUDA training')
    parser.add_argument('--seed', type=int, default='3407', help='random seed')
    parser.add_argument('--dropout', type=float,
                        default=0.5, help='dropout rate')
    parser.add_argument('--workers', default=4, type=int, help='workers')
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
        os.path.join(args.checkpoint, "test.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)
    printf(f"args: {args}")
    printf('==> Building model..')

    net = PointCAT().to(device)
    criterion = cal_loss  # cross_entropy

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    printf(f"Resuming best checkpoint from {args.checkpoint}")
    checkpoint_path = os.path.join(args.checkpoint, "best_checkpoint.pth")
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['net']

    net.load_state_dict(state_dict, strict=False)
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():

        for batch_idx, (data, label) in enumerate(test_loader):
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
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)

        test_acc = float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred)))

        test_acc_avg = float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred)))

        test_loss = float("%.3f" % (test_loss / (batch_idx + 1)))

        printf(
            f"Testing loss:{test_loss} acc_avg:{test_acc_avg}% "
            f"acc:{test_acc}% time:{time_cost}s [best test acc: {test_loss}%] \n\n")
