# NN

This project implements a Neural Network from scratch in C++. It includes functionality to work with the MNIST dataset for handwritten digit classification.

## Project Structure

- **NeuralNetwork.h/cpp**: Core neural network implementation from scratch, including classes for the network, layers, neurons, and connections.
- **MNIST.h/cpp**: Handles loading and processing of the MNIST dataset for use with our C++ neural network.
- **main.cpp**: Entry point for the application, contains training and evaluation logic.
- **NN.py**: TensorFlow implementation for performance comparison with our C++ implementation.

TO-DO:
-
- Add ```ReadTrainingImages(string path)``` - Done
- Add ```ReadTrainingLabels(string path)``` - Done
- Add Training Dataset to ```./MNIST``` - Done
- Add Activation Function - Done
- Add Biases - Done
- Change Ativation Function from Sigmoid to Tanh - Done
- Add ```ForwardPropagate()``` - Done
- Finish refactored code (Weights and Biases in Neuron Class) - Done
- Finish ```NNLayer::CalculateDeltas``` -Done
- Finish ```NNLayer::UpdateWeights``` - Done
- See if ```NNLayer::BackPropagate``` is needed - Done (not needed)
- Change Datasets (Images and TrainingImages) from double to float
- Implements Xavier/Glorot initialization

Non-Priority:
-
- Add catches and error handling
- Nice print statements for Weights


Sugestões:
-
- Trocar para ReLu
- SoftMax para a última
- Testar os outputs e passsos intermédios com ajuda de python, que tem forma de controlar os pesos e propagação(!!)
- Andrew Ng