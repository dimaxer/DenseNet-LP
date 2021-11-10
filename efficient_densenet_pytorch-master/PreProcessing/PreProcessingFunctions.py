from PreProcessing.source import *


# graphTest = nx.Graph()
# graphTest.add_node(0)
# graphTest.add_node(1)
# graphTest.add_node(2)
# graphTest.add_node(3)
# graphTest.add_node(4)
# graphTest.add_edge(0, 1)
# graphTest.add_edge(1, 2)
# graphTest.add_edge(2, 3)
# graphTest.add_edge(3, 4)
# #
# #
# adj=graphTest.adjacency()
# dictionaryTest = dict(nx.all_pairs_shortest_path_length(graphTest))
# iterator = (nx.all_pairs_shortest_path_length(graphTest))
# for i in iterator:
#     print(i)
#
# print("after")
#
# min_dist = 5
# size = 4


def create_node2vec_feature_vectors_to_file(graph, file_name):
    # Precompute probabilities and generate walks - **ON WINDOWS ONLY WORKS WITH workers=1**
    node2vec = Node2Vec(graph, dimensions=dimensionOfFeatures, walk_length=30, num_walks=200, workers=4)  # Use temp_folder for big graphs

    # Embed nodes
    model = node2vec.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `dimensions` and `workers` are automatically passed (from the Node2Vec constructor)

    # Look for most similar nodes
    model.wv.most_similar('2')  # Output node names are always strings

    # Save embeddings for later use
    model.wv.save_word2vec_format(file_name)

"extract a subgraph for a specific node which all subsequesnt node are <= to that distance"
def subgraph_extraction(min_dist, length_dictionary):
    size = len(length_dictionary)
    list_of_min_dist_list = []
    for i in range(0, size):
        min_dist_list = []
        for j in range(0, size):
            # print("length between ", i, " and ", j, " is: ", dictionary[i][j])
            try:
                if length_dictionary[i][j] <= min_dist and i != j:
                    min_dist_list.append(j)
            except KeyError:
                k = 3
        # print("min_dist_list at ", i, " :", min_dist_list)

        list_of_min_dist_list.append(min_dist_list)
    return list_of_min_dist_list

"parse string to int"
def parsing_int(row):
    index = 0
    node_string = ""
    while row[index] != ' ':
        node_string += row[index]
        index = index + 1
    node = int(node_string)
    return node, index + 1

"parse string to double"
def parsing_double(row, index):
    float_string = ""
    try:
        while row[index] != ' ' or index == len(row):
            float_string += row[index]
            index = index + 1
    except IndexError:
        k = 5
    return float(float_string), index + 1


# result = subgraph_extraction(1, dictionaryTest)
# print(result)
# node1 = result[2]
# print(node1)

"return a feature matrix by reading the values from a text file"
def txtFileToMatrix(file_name):
    feature_vector_matrix = np.zeros((N, dimensionOfFeatures))
    with open(file_name) as FILE:
        first_line = 0
        for vector in FILE:
            if first_line == 0:
                first_line = 1
            else:
                node_number, row_index = parsing_int(vector)
                for i in range(dimensionOfFeatures):
                    feature_vector_matrix[node_number][i], row_index = parsing_double(vector, row_index)
        FILE.close()
    return feature_vector_matrix



# print("blbalbla")
# print(featureMatrix[82])
# print(featureMatrix[49])

"create a feature vecor matrix for the subgraph for a specific node"
def subgraph_feature_vector_matrix(node_number, feature_matrix, k, distance_sub_graph, distance_dictionary):
    subGraphFeatureMatrix = np.zeros((k, dimensionOfFeatures))
    subgraph_for_all_nodes = subgraph_extraction(distance_sub_graph, distance_dictionary)
    subgraph_list_for_node = subgraph_for_all_nodes[node_number]
    for i in range(k):
        for j in range(dimensionOfFeatures):
            if i < len(subgraph_list_for_node):
                subGraphFeatureMatrix[i][j] = feature_matrix[subgraph_list_for_node[i]][j]
    return subGraphFeatureMatrix


"Calculate the cosine distance of the sub graph feature vector matrix and return a sorted list of the distance and the node number as a tuple"
def cosine_distance_similarity(matrix, feature_vector):
    sortedFeatureMatrix = np.zeros((K, dimensionOfFeatures))
    i = 0
    list = []
    for vector in matrix:
        num_array = cosine_similarity(vector.reshape(1, -1), feature_vector.reshape(1, -1))
        list.append((num_array[0][0], i))
        i += 1
    list.sort()

    return list

"return the index from the list of tuples that has cosine distane and node number as a tuple"
def list_index(similarity_list, matrix_index):
    index = 0
    for list_tuple in similarity_list:
        if list_tuple[1] == matrix_index:
            return index, list_tuple[0]
        index += 1

"create and return a sorted feature vector matrix"
def sorted_similarity_vectors(similarity_list, subGraph_feature_matrix):
    sorted_feature_matrix = np.zeros((K, dimensionOfFeatures))
    matrix_index = 0
    for vector in subGraph_feature_matrix:
        list_tuple = list_index(similarity_list, matrix_index)
        cos_distance = list_tuple[1]
        if cos_distance != 0:
            tuple_index = list_tuple[0]
            sorted_feature_matrix[tuple_index] = vector
        matrix_index += 1
    return sorted_feature_matrix


# create a list of node pairs for example [(node_number1, node_number2),(node_number3, node_number4)]
def generateNodesCombiantions():
    nodes=list(range(0,N))
    return list(it.combinations(nodes, 2))


# this function creates a (2 x K x K)  matrix for a node pair
def createPairFeaturesMatrix(nodes, FeatureMatrix):
    nodeFeatureMatrixPair = np.zeros((2, K, dimensionOfFeatures))
    i = 0
    for node in nodes:
        subGraphFeatureVectorMatrix = subgraph_feature_vector_matrix(node, FeatureMatrix, K, distance, dictionary)
        cosine_distance_list = cosine_distance_similarity(subGraphFeatureVectorMatrix, FeatureMatrix[node])
        sorted_matrix = sorted_similarity_vectors(cosine_distance_list, subGraphFeatureVectorMatrix)
        nodeFeatureMatrixPair[i] = sorted_matrix
        i = i + 1
    return nodeFeatureMatrixPair


# this function creates a 4 dimensional matrix for all node paris size of: (NumberOfNodePairs x 2 x K x K)
def createAllNodePairFeatureMatrices(FeatureMatrix, NodeCombinationList):
    nodePairFeatureMatrices = np.zeros((numberOfNodePairs, 2, K, dimensionOfFeatures))
    for i in range(numberOfNodePairs):
        nodePairFeatureMatrices[i] = createPairFeaturesMatrix(NodeCombinationList[i], FeatureMatrix)
    return nodePairFeatureMatrices


# this function creates a (3 x K x K)  matrix for a node pair in order to simulate a picture like data. 
# the third channel is the averge between the other 2 channel in the matrix
def createPicturePairFeaturesMatrix(nodes, FeatureMatrix):
    nodeFeatureMatrixPair = np.zeros((3, K, dimensionOfFeatures))
    i = 0
    for node in nodes:
        subGraphFeatureVectorMatrix = subgraph_feature_vector_matrix(node, FeatureMatrix, K, distance, dictionary)
        cosine_distance_list = cosine_distance_similarity(subGraphFeatureVectorMatrix, FeatureMatrix[node])
        sorted_matrix = sorted_similarity_vectors(cosine_distance_list, subGraphFeatureVectorMatrix)
        nodeFeatureMatrixPair[i] = sorted_matrix
        i = i + 1
        for l in range(K):
            for p in range(dimensionOfFeatures):
                nodeFeatureMatrixPair[2][l][p] = (nodeFeatureMatrixPair[0][l][p] + nodeFeatureMatrixPair[1][l][p])/2.0
    return nodeFeatureMatrixPair

# this function creates a 4 dimensional matrix for all node paris (after recreating them as picture like) size of: (NumberOfNodePairs x 3 x K x K)
def createPictureAllNodePairFeatureMatrices(FeatureMatrix, NodeCombinationList):
    nodePairFeatureMatrices = np.zeros((numberOfNodePairs, 3, K, dimensionOfFeatures))
    for i in range(numberOfNodePairs):
        nodePairFeatureMatrices[i] = createPicturePairFeaturesMatrix(NodeCombinationList[i], FeatureMatrix)
    return nodePairFeatureMatrices

# this function creates all true lables for all the node pairs. 
# for example: label 0 means a node pair dosent have a link the original graph
# label 1 means a node pair has a link the original graph
def createNodeEdgeLabels(NodeCombinationList):
    list = []
    for nodePair in NodeCombinationList:
        if original_graph.has_edge(nodePair[0],nodePair[1]):
            list.append(1)
        else:
            list.append(0)
    return list

def save_data_into_file(final_data, file_name):
    np.save(file_name,final_data)

def load_data_from_file(training_file_name):
    return np.load(training_file_name)


def remove_percentage_of_edges_from_graph(graph, percentage_to_remove):
    graph_to_return = graph
    graph_to_return.remove_edges_from(random.sample(graph.edges(),k=int(percentage_to_remove*graph.number_of_edges())))
    return graph_to_return


def create_data_set_to_file(graph, node2vec_feature_vector_file_name, nodeCombinationsList, data_set_file_name):
    create_node2vec_feature_vectors_to_file(graph, node2vec_feature_vector_file_name)
    #create feature matrix for all the nodes in the graph
    node_pair_feature_matrix = txtFileToMatrix(node2vec_feature_vector_file_name)
    all_node_pairs_feature_matrix = createPictureAllNodePairFeatureMatrices(node_pair_feature_matrix, nodeCombinationsList)
    # print(finalMatrix)
    save_data_into_file(all_node_pairs_feature_matrix, data_set_file_name)