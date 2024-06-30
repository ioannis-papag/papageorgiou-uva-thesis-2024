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
