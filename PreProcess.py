import os.path
import traceback
from tkinter import StringVar

from colorama import Fore
from networkx.classes.function import number_of_nodes

import DenseNetGui
from PreProcessingFunctions import *
import source
import networkx as nx
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE




def runPreProcess(graph_file_name):
    try:
        string_to_print = "Loading graph from file: " + graph_file_name
        print(string_to_print)
        DenseNetGui.update_output_PreProccessing(string_to_print)
        original_graph = nx.read_edgelist(source.graph_file_name, create_using=nx.Graph(), nodetype=int)
        num_of_nodes = original_graph.number_of_nodes()
        num_of_node_pairs = int((num_of_nodes * (num_of_nodes - 1)) / 2)
        DenseNetGui.init_progress_bar(int(num_of_node_pairs * 2))
        negative_class_edges = percentage_of_complemant_graph_edges(original_graph)
        # save number of nodes into txt file
        # os.chdir('../')
        f = open(source.num_of_nodes_file_name, "w")
        f.write(str(num_of_nodes))
        f.close()

        #create directory for saving the grpah images { OrginalGraph.png,TestGraph.png,TrainingGraph.png
        GraphImagesPath=os.path.join(source.MainPath,'GraphImages')
        if not os.path.isdir(GraphImagesPath):
            os.mkdir(GraphImagesPath)

        # print loaded graph info
        # create all node combinations list for all possible node pairs so that the model would be able to ask if there is a link between 2 nodes or not
        node_pairs_list = generate_node_combinations(num_of_nodes)
        string_to_print = "Number of node pairs " + str(len(node_pairs_list))
        print(string_to_print)
        DenseNetGui.update_output_PreProccessing(string_to_print)

        # to simulate link prediction we create a graph without 10% of the edges and graph with only those 10% + 80% negative class edges for the training set and testing set respectively
        # create a graph without 10% deleted edges from the orginal graph for the training set
        training_set_graph, removed_edges = remove_percentage_of_edges_from_graph(original_graph,
                                                                                  source.percentage_of_edges_to_remove)

        string_to_print = "Training Set Graph after deleting:"
        print(string_to_print)
        DenseNetGui.update_output_PreProccessing(string_to_print)
        string_to_print= str(
            percentage_of_edges_to_remove) + "% of the edges " + str(nx.info(training_set_graph))
        print(string_to_print)
        DenseNetGui.update_output_PreProccessing(string_to_print)


        # create a graph with only those 10% deleted edges from the orginal graph for the test set
        test_set_graph = nx.Graph()
        test_set_graph.add_nodes_from(original_graph)
        test_set_graph.add_edges_from(removed_edges, color='red')
        test_set_graph.add_edges_from(negative_class_edges,color='black')



        string_to_print = "Test Set Graph with the deleted:"
        print(string_to_print)
        DenseNetGui.update_output_PreProccessing(string_to_print)
        string_to_print = str(
            source.percentage_of_edges_to_remove) + "% edges and addition of "
        print(string_to_print)
        DenseNetGui.update_output_PreProccessing(string_to_print)
        string_to_print= str((source.percentage_of_negative_class_additions - source.percentage_of_edges_to_remove)) +"% random negative class edges" + str(nx.info(test_set_graph))
        print(string_to_print)
        DenseNetGui.update_output_PreProccessing(string_to_print)



        # create graph dictionary for shortest paths
        shortest_path_dict_training_set = dict(nx.all_pairs_shortest_path_length(training_set_graph))
        shortest_path_dict_test_set = dict(nx.all_pairs_shortest_path_length(test_set_graph))

        train_lables = create_pair_labels(node_pairs_list, training_set_graph)
        save_data_into_file(source.train_labels_file_name, train_lables)

        test_labels = create_pair_labels(node_pairs_list, test_set_graph)
        save_data_into_file(source.test_labels_file_name, test_labels)

        # create training set data files
        create_data_set_to_file(training_set_graph, source.training_set_node2vec_feature_vector_file_name, node_pairs_list,
                                source.training_data_file_name, shortest_path_dict_training_set, num_of_nodes
                                )

        # create test set data files
        create_data_set_to_file(test_set_graph, source.test_set_node2vec_feature_vector_file_name, node_pairs_list,
                                source.test_data_file_name, shortest_path_dict_test_set, num_of_nodes)
        num_of_node_pairs = int((num_of_nodes * (num_of_nodes - 1)) / 2)
        DenseNetGui.update_progress_bar(int(num_of_node_pairs * 2), 1)
        # os.chdir('Gui')


    except Exception as e:
        print("Error while run pre processing" +str(e))
        print(Fore.RED + traceback.format_exc() + Fore.RESET)
        DenseNetController.printError(e)


