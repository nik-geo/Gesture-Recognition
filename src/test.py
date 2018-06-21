import argparse
import os
import sys
import shutil
import time
import json
import glob
import signal
import pickle

import numpy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from pprint import pprint
from data_loader_gulpio import VideoFolder
from callbacks import PlotLearning, MonitorLRDecay, AverageMeter
from model import ConvColumn
from torchvision.transforms import *

parser = argparse.ArgumentParser(
    description='PyTorch Jester Training using GulpIO')
parser.add_argument('--config', '-c', help='json config file path')
parser.add_argument('--eval_only', '-e',
                    help="evaluate trained model on validation data.")
parser.add_argument(
    '--resume', '-r', help="resume training from given checkpoint.")
parser.add_argument('--gpus', '-g', help="gpu ids for use.")

args = parser.parse_args()
if len(sys.argv) < 2:
    parser.print_help()
    sys.exit(1)

gpus = [int(i) for i in args.gpus.split(',')]
print("=> active GPUs: {}".format(args.gpus))
best_prec1 = 0

# load config file
with open(args.config) as data_file:
    config = json.load(data_file)


def test():
    # adds a handler for Ctrl+C
    def signal_handler(signal, frame):
        """
        Remove the output dir, if you exit with Ctrl+C and
        if there are less then 3 files.
        It prevents the noise of experimental runs.
        """
        num_files = len(glob.glob(save_dir + "/*"))
        if num_files < 1:
            shutil.rmtree(save_dir)
        print('You pressed Ctrl+C!')
        sys.exit(0)
    # assign Ctrl+C signal handler
    signal.signal(signal.SIGINT, signal_handler)
    # create model
    model = ConvColumn(config['num_classes'])

    # multi GPU setting
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    #Loading the model
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(config['checkpoint'])
    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
        .format(config['checkpoint'], checkpoint['epoch']))

    transform = Compose([
        ToPILImage(),
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    #Load the Data
    val_data = VideoFolder(root=config['val_data_folder'],
                           csv_file_input=config['val_data_csv'],
                           csv_file_labels=config['labels_csv'],
                           clip_size=config['clip_size'],
                           nclips=1,
                           step_size=config['step_size'],
                           is_val=True,
                           transform=transform,
                           )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size= 1, shuffle=False,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=False)


    validate(val_loader, model, criterion)






def validate(val_loader, model, criterion, class_to_idx=None):

    # switch to evaluate mode
    model.eval()
    count=0
    for i, (input, target) in enumerate(val_loader):

        input_vars = torch.autograd.Variable(input.cuda(), volatile=True)
        target_var = torch.autograd.Variable(target.cuda(async=True), volatile=True)

        # compute output and loss
        output = model(input_vars)
        output = output.cpu()
        values,indices = torch.max(output,1)
        opt = int(indices)
        getlable(opt,0)	
        #print(" TARGET VAR - {} ".format(
        #target_var))
        tar = int(target_var)
        getlable(tar,1)


def getlable(idx,ipop):

	switcher = {
		1: "Swiping Left",
		2: "Swiping Right",
		3: "Swiping Down",
		4: "Swiping Up",
		5: "Pushing Hand Away",
		6: "Pulling Hand In",
		7: "Sliding Two Fingers Left",
		8: "Sliding Two Fingers Right",
		9: "Sliding Two Fingers Down",
		10: "Sliding Two Fingers Up",
		11: "Pushing Two Fingers Away",
		12: "Pulling Two Fingers In",
		13: "Rolling Hand Forward",
		14: "Rolling Hand Backward",
		15: "Turning Hand Clockwise",
		16: "Turning Hand Counterclockwise",
		17: "Zooming In With Full Hand",
		18: "Zooming Out With Full Hand",
		19: "Zooming In With Two Fingers",
		20: "Zooming Out With Two Fingers",
		21: "Thumb Up",
		22: "Thumb Down",
		23: "Shaking Hand",
		24: "Stop Sign",
		25: "Drumming Fingers",
		26: "No gesture",
		27: "Doing other things"
	}

	res = switcher.get(idx,'NO GESTURE')
	if(ipop == 0):
		print("OUTPUT - {} ".format(res))
	if(ipop == 1):
		print("TARGET - {} ".format(res))


if __name__ == '__main__':
    test()
