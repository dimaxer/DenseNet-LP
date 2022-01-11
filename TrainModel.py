import signal
import traceback

import fire
import os
import time
import torch
import numpy as np
import networkx as nx
import webbrowser
import subprocess

from colorama import Fore
import tensorboard

import source
import DenseNetGui
import DenseNetController
from models import DenseNet
import PreProcessingFunctions as PFunctions
from torch.utils.tensorboard import SummaryWriter
import PreProcess

def read_num_of_nodes():
    # read number of nodes in a graph
    try:
        num_nodes_file = open(source.num_of_nodes_file_name, 'r')
    except FileNotFoundError as FILE_ERROR:
        fileName = str(FILE_ERROR.filename)
        print('\n\tSorry, \'', fileName, '\' not found.\n')
        print(Fore.RED + traceback.format_exc() + Fore.RESET)
        DenseNetController.printError(
            "file/directory: " + os.path.basename(fileName) + " not found.\nPlease import the all the files")
    lines = num_nodes_file.readlines()
    str_num = ""
    for line in lines:
        for c in line:
            if c.isdigit() == True:
                str_num = str_num + c

    num_nodes_file.close()
    num_of_nodes = int(str_num)
    print("num_of_nodes = ", num_of_nodes)
    return num_of_nodes


def init():
    global batch_size_divider,writer
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_of_nodes = read_num_of_nodes()
    num_of_node_pairs = int((num_of_nodes * (num_of_nodes - 1)) / 2)
    batch_size_divider = round(num_of_node_pairs / source.batch_size)
    # this is for clearing cuda memorry
    torch.cuda.empty_cache()
    writer = SummaryWriter()


class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, train_batch_list, optimizer, epoch, n_epochs, train_labels, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, batch in enumerate(train_batch_list):

        torchLabels = torch.cuda.LongTensor(train_labels[batch_idx])
        torchMatrix = torch.cuda.FloatTensor(batch)
        input = torchMatrix
        target = torchLabels

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]    ' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]    ' % (batch_idx + 1, len(train_batch_list)),
                ' Time %.3f (%.3f)    ' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)    ' % (losses.val, losses.avg),
                '   Error %.4f (%.4f)   ' % (error.val, error.avg),
            ])
            print(res)
            DenseNetGui.updateTrainingoutput(res)


    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, test_batch_list, test_labels, epoch=0, n_epochs=1 ,print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()

    for batch_idx, batch in enumerate(test_batch_list):

        torchLabels = torch.cuda.LongTensor(test_labels[batch_idx])
        torchMatrix = torch.cuda.FloatTensor(batch)
        input = torchMatrix
        target = torchLabels

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Test:   [%d/%d]    ' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]    ' % (batch_idx + 1, len(test_batch_list)),
                ' Time %.3f (%.3f)    ' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)    ' % (losses.val, losses.avg),
                '   Error %.4f (%.4f)   ' % (error.val, error.avg),
            ])
            print(res)
            if DenseNetController.getTrainModelStatus():
                DenseNetGui.updateTrainingoutput(res)
            else:
                DenseNetGui.updateLoadoutput(res)



    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, save_path, n_epochs=300,
          lr=0.1, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    # Start log
    with open(os.path.join(save_path, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

    # Train model

    # load training data
    train_set = PFunctions.load_data_from_file(source.training_data_file_name)
    train_batch_list = np.array_split(train_set, batch_size_divider)

    # load test data
    test_set = PFunctions.load_data_from_file(source.test_data_file_name)
    test_batch_list = np.array_split(test_set, batch_size_divider)

    # load training labels
    load_train_labels = PFunctions.load_data_from_file(source.train_labels_file_name)
    training_labels = np.array_split(load_train_labels, batch_size_divider)

    # load test labels
    load_test_labels = PFunctions.load_data_from_file(source.test_labels_file_name)
    testing_labels = np.array_split(load_test_labels, batch_size_divider)

    for epoch in range(n_epochs):
        # PreProcess.runPreProcess(source.Graph_absoloute_path)
        #
        # train_set = PFunctions.load_data_from_file(source.training_data_file_name)
        # train_batch_list = np.array_split(train_set, batch_size_divider)
        #
        # # load test data
        # test_set = PFunctions.load_data_from_file(source.test_data_file_name)
        # test_batch_list = np.array_split(test_set, batch_size_divider)
        #
        # # load training labels
        # load_train_labels = PFunctions.load_data_from_file(source.train_labels_file_name)
        # training_labels = np.array_split(load_train_labels, batch_size_divider)
        #
        # # load test labels
        # load_test_labels = PFunctions.load_data_from_file(source.test_labels_file_name)
        # testing_labels = np.array_split(load_test_labels, batch_size_divider)

        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            train_batch_list=train_batch_list,
            optimizer=optimizer,
            epoch=epoch, train_labels=training_labels,
            n_epochs=n_epochs,
        )

        writer.add_scalar("Loss/Epoch", train_loss, epoch)
        writer.add_scalar("Accuracy/Epoch", 1 - train_error, epoch)

        scheduler.step()
        _, valid_loss, valid_error = test_epoch(
            model=model_wrapper,
            test_batch_list=test_batch_list,
            test_labels=testing_labels,
            is_test=True,
            epoch = epoch,
            n_epochs=n_epochs
        )

        torch.save(model.state_dict(), os.path.join(save_path, 'model.dat'))

        # Log results
        with open(os.path.join(save_path, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))
    writer.flush()
    writer.close()

    # Final test of model on test set
    model.load_state_dict(torch.load(os.path.join(save_path, 'model.dat')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    DenseNetGui.updateTrainingoutput("")
    DenseNetGui.updateTrainingoutput(source.FinalTest)
    print(source.FinalTest)
    DenseNetGui.updateTrainingoutput("")
    test_results = test_epoch(
        model=model,
        test_batch_list=test_batch_list,
        test_labels=testing_labels,
        is_test=True
    )
    _, _, test_error = test_results
    with open(os.path.join(save_path, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    StringToPrint='Final test error: %.4f' % test_error
    print(StringToPrint)
    DenseNetGui.updateTrainingoutput(StringToPrint)

def start_training(depth=100, growth_rate=12, efficient=True,
                   n_epochs=300, seed=None):
    """
    training of a Denset using graph node pairs data converted into pictures data

    Args:

        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)


        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

    # path to save the results
    save_path = source.results_path

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]



    # Models
    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_init_features=growth_rate * 2,
        num_classes=10,
        small_inputs=True,
        efficient=efficient,
    )
    print(model)

    # Print number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print("Total parameters: ", num_params)

    # Make save directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path):
        raise Exception('%s is not a dir' % save_path)

    # Train the model
    train(model=model, save_path=save_path,
          n_epochs=n_epochs, seed=seed)
    print('Done!')


"""
Try out the efficient DenseNet implementation:
python train_mode.py --efficient True  --save <path_to_save_dir>

Try out the naive DenseNet implementation:
python train_model.py --efficient False  --save <path_to_save_dir>

Other args:
    --depth (int) - depth of the network (number of convolution layers) (default 40)
    --growth_rate (int) - number of features added per DenseNet layer (default 12)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 256)
    --seed (int) - manually set the random seed (default None)
"""

def startTrain():
    try:
        init()
        # fire.Fire(start_training)
        n_epoch=DenseNetGui.getEpoches()
        start_training(n_epochs=n_epoch)
        DenseNetGui.updateTrainingoutput("Training is Done !")
        DenseNetController.printInfo("Training is Done !")
        tb = tensorboard.program.TensorBoard()
        tb.configure(bind_all=True, logdir="runs")
        url = tb.launch()
        webbrowser.open(url)
        print("TensorBoard %s started at %s" % (tensorboard.__version__, url))
    except FileNotFoundError as FILE_ERROR:
        fileName = str(FILE_ERROR.filename)
        print('\n\tSorry, \'', fileName, '\' not found.\n')
        print(Fore.RED + traceback.format_exc() + Fore.RESET)
        DenseNetController.printError("file/directory: " + os.path.basename(fileName) + " not found.\nPlease import the all the files")
    except Exception as e:
        print("Error while run Train Model")
        print(Fore.RED + traceback.format_exc() + Fore.RESET)
        DenseNetController.printError("Error while run Train Model:\n" + str(e))



