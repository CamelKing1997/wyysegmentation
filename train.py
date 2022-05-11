import os
from collections import OrderedDict

import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms
from torch.utils.data.dataloader import DataLoader

from dataloaders.easydataset import easydataset

import logging
from sheen import Str, ColoredHandler
import argparse

logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f'./LOG/trainlog.txt')
file_handler.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(ColoredHandler())


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--onnx", default=True, type=bool)
    parser.add_argument("--onnxsim", default=True, type=bool)

    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--resize", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrstep", type=int, default=100)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--Optimizer', type=str, default='SGD')

    parser.add_argument('--backbone', type=str, default='resnet')
    parser.add_argument("--stride", type=int, default=16)

    parser.add_argument('--dataset', type=str, default='dataset/Project/DN2130301/20220421')
    parser.add_argument('--val_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=20)

    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    opt = get_argparser()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    train_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((opt.resize, opt.resize)),
        torchvision.transforms.ColorJitter(0.3),
        torchvision.transforms.RandomHorizontalFlip(0.5),
        torchvision.transforms.RandomVerticalFlip(0.5),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_trans = torchvision.transforms.Compose([
        torchvision.transforms.Resize((opt.resize, opt.resize)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = easydataset(opt, train_trans, split='train')
    val_dataset = easydataset(opt, val_trans, split='val')

    train_dataloader = DataLoader(train_dataset, batch_size=opt.train_batch_size, pin_memory=False, shuffle=True,
                                  num_workers=opt.workers)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.val_batch_size, pin_memory=False, shuffle=True,
                                num_workers=opt.workers)

    logger.info(f"Train Batchsize:{opt.train_batch_size} Validation Batchsize:{opt.val_batch_size}.")
    logger.info(f"Dataset Loaded.")
    logger.info(f'Train Image Count:{len(train_dataset.images)},Validation Image Count:{len(val_dataset.images)}.')
    logger.info(f'Train Iteration Count:{len(train_dataloader)},Validation Iteration Count:{len(val_dataloader)}.')

    from net.deeplab import DeepLab
    from net.segformer import Segformer
    model = DeepLab(num_classes=opt.num_classes)
    modeltype = 'DeepLab'

    logger.info(f'{modeltype} Loaded.')

    torch.backends.cudnn.benchmark = True

    if opt.checkpoint != "":
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        model.load_state_dict(torch.load(opt.checkpoint, map_location=device))
        logger.info(f"Training state restored from {opt.checkpoint}")
    else:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        weights_init(model)

    logger.info(f'Model loaded into MultiGPU.')

    # Set Optimizer
    if opt.Optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    # Set Scheduler
    len_loader = (len(train_dataloader) * opt.epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / len_loader) ** 0.9)

    # Set Loss
    criterion = torch.nn.CrossEntropyLoss()

    c_epoch = 0
    while True:
        c_epoch += 1
        epoch_train_loss = 0
        epoch_train_accuracy = 0
        for iter, (image, label) in enumerate(train_dataloader):
            model.train()

            image = image.to(device)
            label = label.to(device)
            label = torch.squeeze(label).long()

            optimizer.zero_grad()
            predict = model(image)

            iter_train_loss = criterion(predict, label)
            iter_train_loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_train_loss += iter_train_loss.item()

        meanloss = epoch_train_loss / (iter + 1)
        meanaccuracy = epoch_train_accuracy / (iter + 1)
        logger.info(
            f'Epoch:{c_epoch} TrainLoss:{meanloss:.05f} TrainAccuray:{meanaccuracy:.08f} lr:{get_lr(optimizer):.8f}')
        epoch_train_loss = 0
        epoch_train_accuracy = 0

        # Validation
        if c_epoch % opt.val_interval == 0:
            logger.info(f'Epoch:{c_epoch} Start Validation...')
            epoch_val_loss = 0
            epoch_val_accuracy = 0
            model.eval()
            for val_iteration, (image, label) in enumerate(val_dataloader):
                image = image.to(device)
                label = label.to(device)
                predict = model(image)
                iter_val_loss = criterion(predict, label)
                epoch_val_loss += iter_val_loss.item()

                val_accuracy = torch.mean(
                    (torch.argmax(F.softmax(predict, dim=-1), dim=-1) == label).type(torch.FloatTensor))
                epoch_val_accuracy += val_accuracy.item()
            val_meanloss = epoch_val_loss / (val_iteration + 1)
            val_meanaccuracy = epoch_val_accuracy / (val_iteration + 1)
            logger.info(f'Epoch:{c_epoch} ValidationLoss:{val_meanloss:.05f} ValidationAccuray:{val_meanaccuracy:.08f}')

        # save pt&onnx
        if c_epoch % opt.save_interval == 0:
            ptname = f'{modeltype}_E{c_epoch}_A{(val_meanaccuracy * 100):.08f}.pt'
            torch.save(model.state_dict(), f'LOG/{ptname}')
            logger.info(f'Epoch:{c_epoch} Model save as \'LOG/{ptname}\'')
            if opt.onnx:
                pytorchmodel = model
                pytorchmodel.eval()
                onnxpath = ptname[:-3] + '.onnx'
                onnxsimpath = ptname[:-3] + '_sim.onnx'
                im = torch.rand((1, 3, opt.resize, opt.resize))
                onnxmodel = pytorchmodel
                checkpoint = torch.load(ptname)
                new_checkpoint = OrderedDict()
                for k, v in checkpoint.items():
                    name = k[7:]
                    new_checkpoint[name] = v
                onnxmodel.load_state_dict(new_checkpoint)
                logger.info(f"Model restored from {opt.checkpoint}")
                del checkpoint
                del pytorchmodel
                del onnxmodel

                torch.onnx.export(onnxmodel, im, onnxpath, verbose=False, opset_version=13,
                                  training=torch.onnx.TrainingMode.EVAL,
                                  do_constant_folding=True,
                                  input_names=['image'],
                                  output_names=['output'])
                logger.info(f'Epoch:{c_epoch} Onnx model save as \'{onnxpath}\'.')
                if opt.onnxsim:
                    model_onnx = onnx.load(onnxpath)
                    onnx.checker.check_model(model_onnx)
                    model_onnx, check = onnxsim.simplify(model_onnx)
                    if not check:
                        logger.info('onnx check failed')
                    else:
                        onnx.save(model_onnx, onnxsimpath)
                        logger.info(f'Epoch:{c_epoch} Simplified onnx model save as \'{onnxsimpath}\'.')

        if c_epoch == opt.epoch:
            break


def weights_init(net, init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    logger.info('Initialize network with kaiming type')
    net.apply(init_func)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    main()
