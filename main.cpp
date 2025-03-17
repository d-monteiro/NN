#include <iostream>
#include <chrono>
#include <vector>
#include <random>

#include "MNIST.h"
#include "NeuralNetwork.h"

using namespace std;

//g++ main.cpp MNIST.cpp NeuralNetwork.cpp -o main && ./main

int main(){

    // PART 1: MNIST Loading
    Reader reader;

    auto start = chrono::high_resolution_clock::now();
    reader.ReadImages("./MNIST/train-images.idx3-ubyte");
    reader.ReadLabels("./MNIST/train-labels.idx1-ubyte");

    reader.ReadTrainingImages("./MNIST/t10k-images.idx3-ubyte");
    reader.ReadTrainingLabels("./MNIST/t10k-labels.idx1-ubyte");
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> loading_time = end - start;
    cout << "Total MNIST Loading time: " << loading_time.count() << " seconds" << endl; //Loading time: 4.1225 seconds

    reader.PrintImage(3502);
    cout << "Label of image 3502: ";
    reader.PrintLabel(3502);

    cout << endl;

    reader.PrintTrainingImage(9999);
    cout << "Label of training image 9999: ";
    reader.PrintTrainingLabel(9999);

    // PART 2: Neural Network Initialization
    vector<int> nNeurons = {784, 128, 10};

    auto start1 = chrono::high_resolution_clock::now();
    NeuralNetwork nn(nNeurons);
    auto end1 = chrono::high_resolution_clock::now();
    chrono::duration<double> loading_time1 = end1 - start1;
    
    cout << endl;
    
    cout << "Neural network created with " << nNeurons.size() << " layers:" << endl;
    for (size_t i = 0; i < nn.Layers.size(); ++i) {
        cout << "Layer " << i << " has " << nn.Layers[i]->Neurons.size() << " neurons." << endl;
    }
/*  
    cout << endl;
    
    auto start2 = chrono::high_resolution_clock::now();
    nn.Initialize();
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> loading_time2 = end2 - start2;

    cout << "Creation time: " << loading_time1.count() << " seconds" << endl;
    cout << "Initialization time: " << loading_time2.count() << " seconds" << endl;
    cout << "Total time: " << loading_time1.count() + loading_time2.count() << endl;

    // PART 3: Forward Propagation
    vector<double> sampleInput(784, 0.5); // Example input with all values set to 0.5
    nn.Input = sampleInput;
    
    auto start3 = chrono::high_resolution_clock::now();
    nn.ForwardPropagate();
    auto end3 = chrono::high_resolution_clock::now();
    chrono::duration<double> loading_time3 = end3 - start3;

    cout << "FeedForward time: " << loading_time3.count() << " seconds" << endl;

    cout << endl;

    cout << "Output of the last layer:" << endl;
    for (size_t i = 0; i < nn.Layers.back()->Neurons.size(); ++i) {
        cout << nn.Layers.back()->Neurons[i]->Output << " ";
    }
    cout << endl;
*/
    return 0;
}