import traceback

from colorama import Fore

import DenseNetController
import DenseNetGui
from models import DenseNet
import torch
import os
from TrainModel import test_epoch
from TrainModel import read_num_of_nodes
import PreProcessingFunctions as PFunctions
import numpy as np
import source
def RunLoadModel():
    # os.chdir('../')

    num_of_nodes = read_num_of_nodes()
    num_of_node_pairs = int((num_of_nodes*(num_of_nodes - 1))/2)
    batch_size_divider = round(num_of_node_pairs/source.batch_size)

    # load training data
    train_set = PFunctions.load_data_from_file(source.training_data_file_name)
    test_batch_list = np.array_split(train_set, batch_size_divider)

    #load training labels
    load_train_labels = PFunctions.load_data_from_file(source.train_labels_file_name)
    test_labels = np.array_split(load_train_labels, batch_size_divider)

    model_path = os.path.join(source.results_path,'model.dat')

    depth=100
    growth_rate=12
    efficient=True
    n_epochs=300
    seed=None
    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(3)]

    model = DenseNet(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=growth_rate*2,
            num_classes=10,
            small_inputs=True,
            efficient=efficient,
        )

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()
    try:
        model.load_state_dict(torch.load(model_path))
    except FileNotFoundError as FILE_ERROR:
        fileName = str(FILE_ERROR.filename)
        print('\n\tSorry, \'', fileName, '\' not found.\n')
        print(Fore.RED + traceback.format_exc() + Fore.RESET)
        DenseNetController.printError(
            "file/directory: " + os.path.basename(fileName) + " not found.\nPlease import the all the files")
        return
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
           model = torch.nn.DataParallel(model).cuda()


    test_results = test_epoch(
        model=model,
        test_batch_list=test_batch_list,
        test_labels = test_labels,
        is_test=True
    )
    _, _, test_error = test_results

    StringToPrint="test error: "+ str(test_error)
    print(StringToPrint)
    DenseNetGui.updateLoadoutput(StringToPrint)
