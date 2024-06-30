# Learning to predict Grounded Logical Predicates from Visual Representations

This work is support the thesis conducted for the MSc Information Studies (Data Science track).

The method of AI Planning allows for solving complex problems through representations of their environment using logical predicates. In many domains, however, there is no direct access to predicates but only to complex sensory inputs such as images. This study aims to design a model for automated predicate extraction from images, which constitutes an end-to-end system containing a Convolutional Neural Network (CNN), a Multilayer Perceptron (MLP) and a Graph Neural Network GNN. These components are used for object detection, encoding and predicate prediction respectively. To measure the performance of the model, two datasets consisting of the BlocksWorld game and a more complex variation of it enriched with the Fashion MNIST Dataset were used. The proposed model is tested against both its ablated version, where each component is trained individually, as well as a state-of-the-art model. Throughout the experiments, the end-to-end approach managed to outperform the individually trained model, while achieving results comparable to that of the state-of-the-art model with simpler inputs.

## Setting up the Environment



