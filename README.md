# Rumor-Detection
The public code for [paper](https://www.researchgate.net/publication/344712414_A_Graph_Convolutional_Encoder_and_Decoder_Model_for_Rumor_Detection) **A Graph Convolutional Encoder and Decoder Model for Rumor Detection** which is accepted by DSAA 2020

# Table of Contents
- data <br>
After decompress **data.rar**, you can get three folds named *Twitter15*,*Twitter16*, *Weibo*. Each directory contains two types of file: feature file and label file.<br>
For feature file, it's a delimited file using '\t', which includes information such as 'eid', 'indexP', 'indexC', 'max_degree', 'maxL' and 'Vec'.

		eid: root id
		indexP: index of parent
		indexC: index of current
		max_degree: the total number of the parent node in the tree
		maxL: the maximum length of all the texts from the tree
		Vec: list of index and count
For label file, every root id corresponds a label.
- Process <br>
	- getTwittergraph.py <br>
		To deal with feature file and record the relationship between each node. Meanwhile, store the feature matrix of each node. Finally save all the information into file as '.npy' format.
	- getWeibograph.py <br>
		Done same operation as *getTwittergraph.py*
	- rand5fold.py <br>
		To deal with label file and generate 5-fold lists for valid-set and train-set.
	- process.py <br>
		To define an own PyG graph dataset to get batchsize of data.
- tools <br>
	- earlystopping.py <br>
		In the experiment, we set patience equal to 10, that means when the score doesn't improve for 10 iterations, we will early stop training and save the model result.
	- earlystopping2class.py <br>
		Done same operation as *earlystopping.py* but for Weibo dataset.
	- evaluate.py <br>
		Define some criteria like accuracy and F1 score.
- model <br>
	- GAE.py &nbsp; 
		Our base model using GAE as Decoder Module
	- VGAE.py &nbsp;
		Our base model using VGAE as Decoder Module
	- only_gcn.py &nbsp;
		Comparative trial
	- MVAE.py &nbsp;
		Comparative trial
	- add_root_info.py &nbsp;
		Trick to enhance better representation of data
	- base_BU.py &nbsp;
		Reverse the data flow
	- bidirect.py &nbsp;
		Try to use two directions of data flow
	- Model_Twitter.py &nbsp;
		Main function to run on Twitter
	- Model_Weibo.py &nbsp;
		Main function to run on Weibo
# Experiment
We implement our models using the same set of hyper parameters in our experiments. The batch size is 128. The hidden dim is 64. The total process is iterated upon 50 epochs. The learning rate is 5e-4. We randomly split the datasets and conduct a 5-fold cross-validation and use *acc.* and *f1* as criteria.
### Quick start
#### Step1: Prepare Data
After decompress *data.rar*, using command
    	
	python getTwittergraph.py

#### Step2: Train Model
With two arguments, first stands for dataset's name, the latter is the name of the model ('GCN','GAE','VGAE' can be chosen)

	python Model_Twitter.py Twitter15 VGAE
# Result
Here we only show part of result in the experiment, more details can be seen in the paper.

model_name \ acc. | Twitter | Weibo |
:-: | :-: | :-: |
baseline | 0.737 | 0.908 |
only GCN | 0.840 | 0.935 |
AE-GCN | 0.851 | 0.942 |
VAE-GCN | **0.856** | **0.944** |

Except the main experiment, we also try some tricks to improve model, however we get the worse effect.

model_name | result |
:-: | :-: |
only GCN | 0.8396 |
one-layer GCN | 0.8498 |
two-layers GCN | 0.8367 |
GAT | 0.7879 |
GCN add root | 0.7374 |
bidirect | 0.8294 |
GAE | 0.8498 |
Bottom-up direction GAE | 0.3535 |
