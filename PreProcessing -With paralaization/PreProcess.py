import torch
from torch_geometric.nn import Node2Vec
from PreProcessingFunctions import *


graph_with_deleted_edges = remove_percentage_of_edges_from_graph(original_graph, percentage_of_edges_to_remove)

#create all node combinations
nodeCombinationsList = generateNodesCombiantions()
trueLables = createNodeEdgeLabels(nodeCombinationsList)

create_data_set_to_file(original_graph, original_graph_node2vec_feature_vector_file_name , nodeCombinationsList, training_data_file_name)
create_data_set_to_file(graph_with_deleted_edges, deleted_edges_graph_node2vec_feature_vector_file_name, nodeCombinationsList, test_data_file_name)


# # create training data
# training_data = load_data_from_file(training_data_file_name)
# # print(training_data)
# training_batch_list = np.array_split(training_data,batch_size_divider)
# print(training_batch_list)

# # create test data
# test_data = load_data_from_file(test_data_file_name)
# test_batch_list = np.array_split(test_data, batch_size_divider)
# print(test_batch_list)



