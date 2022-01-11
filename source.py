import os

'''parameters from gui'''
Graph_name=""
Graph_absoloute_path=""
version="v1.0.0"
progress_cnt = 1
FinalTest="--------------------------------------------------Final Test--------------------------------------------------"

K = 64
DimensionOfFeatures = 64
distance = 3
batch_size = 16
n_epochs=100
percentage_of_edges_to_remove = 0.1
percentage_of_negative_class_additions = 1 - percentage_of_edges_to_remove

''' Menu params'''
GitHub = 'https://github.com/dimaxer/DenseNet-LP'
GishaText_9="-family {Gisha} -size 9"
about_header_instructions = "Link Prediction Project-Ort Braude\n version : "+ version + "\nNoam Keren\tDimitry yavestigneyev"
about_footer_instructions = ""
TrainModel=True

'''path to the data, both where to save the data and from where to read the data '''
data_path = "data"
# path to save results csv and the model
results_path = "results"
# path for the graph file, labels ,features_vectors , and data_sets, number_of_nodes 
graph_path = "graph"
labels_path = os.path.join(data_path, "labels")



'''file names'''
MainPath=""
training_set_node2vec_feature_vector_file_name = "training_set_features_vector.txt"
test_set_node2vec_feature_vector_file_name = "test_set_features_vector.txt"
training_data_file_name = "train_data_file.npy"
train_labels_file_name = "train_labels.npy"
test_data_file_name = "test_data_file.npy"
test_labels_file_name = "test_labels.npy"
ImageSavePath="GraphImages"
graph_file_name = ""
num_of_nodes_file_name = "number_of_nodes.txt"
feature_vectors_path = os.path.join(data_path, "feature_vectors")
data_set_path = os.path.join(data_path, "data_set")



'''merge filenames with the path'''
def initPath():
    global MainPath, training_set_node2vec_feature_vector_file_name, test_set_node2vec_feature_vector_file_name, training_data_file_name
    global train_labels_file_name, test_data_file_name, test_labels_file_name, ImageSavePath, num_of_nodes_file_name, feature_vectors_path, data_set_path

    feature_vectors_path = os.path.join(MainPath,data_path, "feature_vectors")
    data_set_path = os.path.join(MainPath,data_path, "data_set")
    training_set_node2vec_feature_vector_file_name = os.path.join(MainPath,feature_vectors_path, training_set_node2vec_feature_vector_file_name)
    test_set_node2vec_feature_vector_file_name = os.path.join(MainPath,feature_vectors_path, test_set_node2vec_feature_vector_file_name)
    training_data_file_name = os.path.join(MainPath,data_set_path, training_data_file_name)
    test_data_file_name = os.path.join(MainPath,data_set_path, test_data_file_name)
    train_labels_file_name = os.path.join(MainPath,labels_path, train_labels_file_name)
    test_labels_file_name = os.path.join(MainPath,labels_path, test_labels_file_name)

    num_of_nodes_file_name = os.path.join(MainPath,'data', num_of_nodes_file_name)
    ImageSavePath = os.path.join(MainPath, "GraphImages")

def set_graph_file_name(graph_name):
    global graph_file_name,Graph_name
    graph_file_name = os.path.join(MainPath, graph_path, graph_name)
    Graph_name=graph_name
