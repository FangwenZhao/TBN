import configargparse
import data_loader
import os
import torch
import models
import utils
from digit_data_loader import load_digit_data
from utils import str2bool
import numpy as np
import random


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add("--seed", type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, default='./data/office31/')
    parser.add_argument('--src_domain', type=str, default='amazon')
    parser.add_argument('--tgt_domain', type=str, default='webcam')

    # training related
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument('--early_stop', type=int, default=20, help="Early stopping")
    parser.add_argument('--epoch_based_training', type=str2bool, default=False,
                        help="Epoch-based training / Iteration-based training")
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=0.75)
    parser.add_argument('--transfer_loss', type=str, default='none')
    parser.add_argument('--it', type=int, default=1)


    return parser


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''

    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    if args.backbone == 'lenet':
        source_loader, target_train_loader, target_test_loader = load_digit_data(
            folder_src, folder_tgt, batch_size=args.batch_size)
        n_class = 10
        return source_loader, target_train_loader, target_test_loader, n_class
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class


def get_model(args):
    model = models.TransferNet(
        args.n_class, transfer_loss=args.transfer_loss, base_net=args.backbone, max_iter=args.max_iter,
        use_bottleneck=args.use_bottleneck).to(args.device)
    return model


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 1.0
    params = model.get_parameters(initial_lr=initial_lr)

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=False)
    return optimizer


def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: args.lr * (1. + args.lr_gamma * float(x)) ** (
        -args.lr_decay))
    return scheduler


def test(model, target_test_loader, args):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(args.device), target.to(args.device)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc, test_loss.avg

def train(source_loader, target_train_loader, target_test_loader, model, optimizer, lr_scheduler, args):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch
    save_root = './results/test_v1/' + args.src_domain + '_' + args.tgt_domain \
                + '_' + str(args.seed)
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)

    best_acc = 0
    stop = 0

    for e in range(1, args.n_epoch + 1):
        model.train()
        model.epoch_based_processing(n_batch)

        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)

        correct = 0
        correct1 = 0
        correct2 = 0

        for i in range(n_batch):
            data_source, label_source = next(iter_source)  # .next()
            data_target, label_target = next(iter_target)  # .next()

            data_source, label_source = data_source.to(
                args.device), label_source.to(args.device)

            data_target, label_target = data_target.to(
                args.device), label_target.to(args.device)

            source_loss, features, target_label = model(data_source, data_target, label_source)
            correct += torch.sum(target_label[1] == label_target)

            target_label, labels_tar, target_loss = model(None, features, target_label, args.it)
            correct1 += torch.sum(labels_tar == label_target)
            correct2 += torch.sum(target_label == label_target)

            p = (e * n_batch + i) / (args.n_epoch * n_batch)
            lamb = (2. / (1. + np.exp(- p)) - 1)

            loss = source_loss + lamb * target_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler:
                lr_scheduler.step()

        acc = 100. * correct / (n_batch * args.batch_size)
        acc1 = 100. * correct1 / (n_batch * args.batch_size)
        acc2 = 100. * correct2 / (n_batch * args.batch_size)


        info = 'Epoch: [{:2d}/{}], target_acc: {:.4f},target1_acc: {:.4f},target2_acc: {:.4f}'.format(
            e, args.n_epoch, acc, acc1, acc2)
        # Test
        stop += 1

        test_acc, test_loss = test(model, target_test_loader, args)
        info += ', Test_acc: {:.4f}'.format(test_acc)

        if best_acc < test_acc:
            best_acc = test_acc
            stop = 0
        if best_acc == 100:
            break
            # state = {'model': model.state_dict(),
            #          'optimizer': optimizer.state_dict(),
            #          'epoch': e}
            # torch.save(state, save_root + '_best.pth')
        if args.early_stop > 0 and stop >= args.early_stop:
            break
        print(info)

    print('Transfer result: {:.4f}'.format(best_acc))


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    print(args)

    set_random_seed(args.seed)
    source_loader, target_train_loader, target_test_loader, n_class = load_data(args)

    setattr(args, "n_class", n_class)
    if args.epoch_based_training:
        setattr(args, "max_iter", args.n_epoch * min(len(source_loader), len(target_train_loader)))
    else:
        setattr(args, "max_iter", args.n_epoch * args.n_iter_per_epoch)
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    if args.lr_scheduler:
        scheduler = get_scheduler(optimizer, args)
    else:
        scheduler = None

    train(source_loader, target_train_loader, target_test_loader, model, optimizer, scheduler, args)


if __name__ == "__main__":
    main()
