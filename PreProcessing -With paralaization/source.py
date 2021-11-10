import networkx as nx
from node2vec import Node2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import itertools as it
import random
# import sys
# np.set_printoptions(threshold=sys.maxsize)
N = 100
K = 64
dimensionOfFeatures = 64
numberOfNodePairs = int((N*(N - 1))/2)
distance = 3
batch_size = 64
batch_size_divider = round(numberOfNodePairs/batch_size)

# file names
original_graph_node2vec_feature_vector_file_name = "orignal_graph_features_vector.txt"
deleted_edges_graph_node2vec_feature_vector_file_name = "deleted_edges_graph_features_vector.txt"
training_data_file_name = "training_data_file.npy"
test_data_file_name = "test_data_file.npy"

percentage_of_edges_to_remove = 0.1
# Create a graph
original_graph = nx.fast_gnp_random_graph(n=N, p=0.3)
#create graph dictionary
dictionary = dict(nx.all_pairs_shortest_path_length(original_graph))

# realData = Data(x=AllNodePairFeatureMatrices, y= trueLables)
# dataList = [realData]
# train_set = DataLoader(dataList, batch_size=32)
#
# testData = Data(x=AllNodePairFeatureMatrices, y= trueLables)
# testList = [testData]
# test_set = DataLoader(testList, batch_size=32)