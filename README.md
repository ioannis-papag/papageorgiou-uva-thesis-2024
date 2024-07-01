# Learning to predict Grounded Logical Predicates from Visual Representations

This work is support for the thesis conducted in the context of the MSc Information Studies (Data Science track).

The method of AI Planning allows for solving complex problems through representations of their environment using logical predicates. In many domains, however, there is no direct access to predicates but only to complex sensory inputs such as images. This study aims to design a model for automated predicate extraction from images, which constitutes an end-to-end system containing a Convolutional Neural Network (CNN), a Multilayer Perceptron (MLP) and a Graph Neural Network GNN. These components are used for object detection, encoding and predicate prediction respectively. To measure the performance of the model, two datasets consisting of the BlocksWorld game and a more complex variation of it enriched with the Fashion MNIST Dataset were used. The proposed model is tested against both its ablated version, where each component is trained individually, as well as a state-of-the-art model. Throughout the experiments, the end-to-end approach managed to outperform the individually trained model, while achieving results comparable to that of the state-of-the-art model with simpler inputs.

## Setting up the Environment

All of the experiments utilize Python 3.12

### Creating the datasets

Both datasets are provided in the premade_datasets folder, but new datasets can also be created on demand using the create_blocks_dataset.py and create_mnist_dataset.py files. To do so, both the PDDLGym and Tensorflow libraries need to be installed using the following command:

```sh
pip install pddlgym tensorflow
```

Afterwards, replace the blocks.py file of PDDLGym (found in *path_to_your_pddlgym_installation\pddlgym\rendering\blocks.py*) with the contained in the pddlgym_changes folder. The changes included in this folder ensure the following:
- The colors of the blocks in the case of the BlocksWorld dataset are known from the beginning and remain constant throughout the generation of the dataset
- There is an option to create the Fashion MNIST BlocksWorld dataset using keras
- The clothes of the blocks in the Fashion MNIST case can be deterministically set during each run by passing a generator when calling the *render* function

#### Custom BlocksWorld dataset
To create a custom BlocksWorld dataset, the script create_blocks_dataset.py can be used with the following command line arguments:
- --seed: Must be an integer. Sets the random seed to be followed throughout the creation of the dataset for reproducibility. Default value is 1.
- --length: Must be an integer greater than 0 and dictates the length of the dataset to be generated. The default value is 7000
- --unique_layouts: Either True or False and dictates whether the dataset will only contain unique layouts of the blocks. If the length of the dataset is greater than the maximum number of possible layouts, the length is set to match that number, resulting in a smaller dataset.

#### Custom Fashion MNIST BlocksWorld dataset
To create a custom Fashion MNIST BlocksWorld dataset, the script create_mnist_dataset.py can be used with the following command line arguments:
- --seed: Must be an integer. Sets the random seed to be followed throughout the creation of the dataset for reproducibility. Default value is 1.
- --length: Must be an integer greater than 0 and dictates the length of the training dataset to be generated. The default value is 50000.
- --unique_test: Either True or False and dictates whether the test dataset will only contain unique layouts of the blocks. The default value is True
- --test_length: Must be an integer greater than 0 and dictates the length of the test set to be generated. The default value is 10000.

## Training a model

There are 4 types of models that can be trained:
- An end-to-end model containing all the components
- The Object Detector (CNN)
- The Object Encoder and GNN
- The SORNet Benchmark model

### End-to-End Model training
To train an end-to-end model, the file *run_pipeline_experiment.py* can be used with the following command line arguments:
- --runs: Integer dictacting how many consequtive models will be trained. Each of them is initialized with a different seed, ranging from 1 up to the number of runs provided. Default value: 1
- --dataset: String indicating the type of dataset the model will be trained one. Can take the values: "blocks" or "mnist". Default value: "blocks"
- --model: String indicating the size of model to be trained. Can take the values: "small", "medium", "large". Default value: "small"
- --path: String containing the (absolute) path of the directory containing the dataset to be used for training.

### CNN Model training
To train a CNN Object Detector separately, the file *run_cnn_experiment.py* can be used with the following command line arguments:
- --runs: Integer dictacting how many consequtive models will be trained. Each of them is initialized with a different seed, ranging from 1 up to the number of runs provided. Default value: 1
- --dataset: String indicating the type of dataset the model will be trained one. Can take the values: "blocks" or "mnist". Default value: "blocks"
- --model: String indicating the size of model to be trained. Can take the values: "small", "medium", "large". Default value: "small"
- --path: String containing the (absolute) path of the directory containing the dataset to be used for training.

### Object Encoder and GNN Model Training
To train both the Object Encoder (MLP) and GNN separately, the file *run_gnn_experiment.py* can be used with the following command line arguments:
- --runs: Integer dictacting how many consequtive models will be trained. Each of them is initialized with a different seed, ranging from 1 up to the number of runs provided. Default value: 1
- --dataset: String indicating the type of dataset the model will be trained one. Can take the values: "blocks" or "mnist". Default value: "blocks"
- --model: String indicating the size of model to be trained. Can take the values: "small", "medium", "large". Default value: "small"
- --path: String containing the (absolute) path of the directory containing the dataset to be used for training.

### SORNet Benchmark Model Training
To train the SORNet model, the file *run_sornet_experiment.py* can be used with the following command line arguments:
- --runs: Integer dictacting how many consequtive models will be trained. Each of them is initialized with a different seed, ranging from 1 up to the number of runs provided. Default value: 1
- --dataset: String indicating the type of dataset the model will be trained one. Can take the values: "blocks" or "mnist". Default value: "blocks"
- --path: String containing the (absolute) path of the directory containing the dataset to be used for training.

## Testing the models

To test the trained models, the file *test_model.py* can be used, for all types of models. The directory containing the models should have one of the following structures depending on the type of model:

### Pipelines:
* *parent_directory* This directory contains all the trained models

### Individually trained models:
* *parent_directory*
  * *parent_directory/cnns* This directory contains all the trained CNNs
  * *parent_directory/encoders* This directory contains all the trained Object Encoders
  * *parent_directory/gnns* This directory contains all the trained GNNs

### SORNet:
* *parent_directory*
  * *parent_directory/models* This directory contains all the trained autoencoder models
  * *parent_directory/rnets* This directory contains all the ReadoutNets
 
If these structures are followed, the *test_model.py* can be run using the following command line arguments:
- --model_type: String indicating the type of model to be tested. Can take the values: "small", "medium", "large", "small_individual", "medium_individual", "large_individual", "sornet"
- --dataset: String indicating the type of dataset the model will be trained one. Can take the values: "blocks" or "mnist". Default value: "blocks"
- --model_dir: String indicating the (absolute) path to the model (parent) directory as described above.
- --path: String containing the (absolute) path of the directory containing the dataset to be used for testing. In the case of the BlocksWorld, there should only be one dataset containing all the samples and this file automatically performa a train-validation-test split. In the case of the Fashion MNIST BlocksWorld, the path should point to a directory containing only the training set, as produced by the files in *create_datasets* folder
