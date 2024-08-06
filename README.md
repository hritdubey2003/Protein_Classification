# Protein classification using GIN, GCN and their ensemble


## 1 Introduction:

### 1.1 Impact:
Classifying whether a protein is an enzyme or not can have a significant impact on a wide range of fields, including biochemistry, molecular biology, and drug discovery. Enzymes are proteins that catalyze biochemical reactions in cells, and they play a crucial role in many biological processes, including metabolism, signaling, and gene expression. Identifying enzymes is essential for understanding these processes and developing new treatments for diseases that involve enzymatic dysfunction.

### 1.2 Problem Statement:
The classification of proteins as enzymes plays a crucial role in many biological processes, such as digestion, metabolism, and energy production.  If a protein is classified as an enzyme, it means that it has the ability to speed up a specific chemical reaction, which is essential for many biological processes to occur. Enzymes work by lowering the activation energy required for a reaction to take place, which allows the reaction to occur more quickly and efficiently.


### 1.3 Goal:
Our goal is to apply the GCN, GAT and GIN neural models in order to have a better understanding about the classification of proteins as enzymes. This can ultimately lead to better drug discovery, biological research and evolutionary analysis. This also gives us exposure to dealing with graph data and implementing graph neural networks.

## 2 Work Done

### 2.1 Data collection:
This project’s data is collected from the geometric datasets found using Pytorch. The data set contains many data points related to proteins. The PROTEINS dataset is frequently used in the field of bioinformatics. It consists of 1113 graphs that represent proteins, with amino acids serving as nodes. An edge is present between two nodes when they are within a certain distance of each other (< 0.6 nanometers). The primary objective of this dataset is to determine whether a given protein is an enzyme or not. Enzymes are a type of protein that act as catalysts to speed up various chemical reactions in the cell. They play a vital role in important bodily functions such as digestion (lipases) and respiration (oxidases), as well as commercial applications such as the production of antibiotics.


![alt text](https://github.com/hritdubey2003/Protein_Classification/blob/main/images/stat_table.png)

Fig 1: Data Statistics


### 2.2 Data analysis:

First we take our data set in graph form and get the 3D spring layout. Once we extract the node and edge positions from the graph information we create the 3D figure and plot the data points. The given projection is not the exact representation of the protein structure since the orientation can be extremely complex, but it gives us an idea of how it would look like.

![alt text](https://github.com/hritdubey2003/Protein_Classification/blob/main/images/3d_projection.png)

Fig 2: 3D projection of Protein structure

We then create a seed and begin training the neural network.

### 2.3 Model Comparisons:
Graph isomorphism networks (GINs) are mathematically better than graph convolutional networks (GCNs) because they do not suffer from the problem of over-smoothing that plagues GCNs. In GCNs, information from neighboring nodes is repeatedly aggregated, which leads to a gradual loss of discriminative power and a flattening of the feature representation. This can result in indistinguishable node embeddings for different nodes in the graph, even if they have different roles or properties. This phenomenon is known as over-smoothing and can significantly limit the performance of GCNs. 

![alt text](https://github.com/hritdubey2003/Protein_Classification/blob/main/images/gcn_classification.png)

Fig 3

GINs, on the other hand, use a more flexible aggregation function that is not prone to over-smoothing. The aggregation function in GINs is based on an isomorphism test between the original graph and the sum of the hidden representations at each layer. This allows GINs to better preserve the original graph structure and to capture more fine-grained information about each node.

![alt text](https://github.com/hritdubey2003/Protein_Classification/blob/main/images/gin_classification.png)

Fig 4

 Mathematically, GINs use a learnable permutation-invariant function to aggregate information from the neighborhood of each node. This function takes as input the sum of the hidden representations of the neighboring nodes and the current node, and outputs a new hidden representation for the current node. By using a permutation-invariant function, GINs ensure that the order of the nodes in the neighborhood does not affect the aggregation result, which makes them more robust to variations in the graph topology. 

Overall, the mathematical superiority of GINs over GCNs lies in their ability to capture more fine-grained information about the graph structure and to avoid the problem of over-smoothing, which can significantly limit the performance of GCNs.

### 2.3 Cross Validation Results
We did 10 fold cross validation on all three of the models; GCN, GIN and GCN+GIN.

The results are somewhat similar to those that we presented in the slides during the presentation.

![alt text](https://github.com/hritdubey2003/Protein_Classification/blob/main/images/result.png)

Fig 5: Our results during presentation



![alt text](https://github.com/hritdubey2003/Protein_Classification/blob/main/images/cross_val1.png)

This is the 10 fold cross validation we did that supports the average for GCN+GIN accuracy.



![alt text](https://github.com/hritdubey2003/Protein_Classification/blob/main/images/cross_val2.png)

This is the 10 fold cross validation for GCN and GIN separately which is very close to our GCN and GIN accuracy.



## 3.1 Challenges faced:
We are currently facing challenges training the neural network. It is nothing too obstructive, we just need to apply more time. At the end of the day, time is the real challenge. We found that the more time we spend on the project the better our results are. It is just unfortunate that we cannot afford the time required all week because of our commitments to other classes.

### 3.2 Limitations:
Although GCN and GIN models have shown promising results in protein classification tasks, there are some limitations that need to be considered:

1. Graph size: GCN and GIN models can struggle with larger graphs, which can lead to scalability issues. This is because the size of the graph can affect the size of the adjacency matrix, and thus the computational complexity of the model.

2. Limited expressive power: While GCN and GIN models are capable of capturing local structural features of graphs, they may not be able to capture more complex global structures. This is because they are limited to a fixed number of graph convolutional layers, and may not be able to fully capture complex relationships between nodes in the graph.

3. Lack of interpretability: GCN and GIN models are often described as "black box" models, meaning that it can be difficult to interpret how the model is making its predictions. This can make it challenging to identify the underlying features or factors that are driving the model's decision-making.

It is important to note that these limitations are not unique to GCN and GIN models and are shared by other machine learning models as well. Nonetheless, these limitations highlight some of the challenges and considerations that need to be taken into account when using GCN and GIN models for protein classification.


## 4 Next plan:
In the future, we would like to implement the Graph Attention Network (GAT)  Architecture to determine if the accuracy exceeds that of Graph Convolutional Networks and Graph Isomorphism Network. We would like to combine all three models, GIN, GCN and GAT to make a better prediction model.
