#include <iostream>
#include <chrono>

#include "MNIST.h"
//#include "NeuralNetwork.h"

using namespace std;

//g++ main.cpp MNIST.cpp NeuralNetwork.cpp -o main && ./main

int main(){
    Reader reader;

    auto start = chrono::high_resolution_clock::now();
    reader.ReadImages("./MNIST/train-images.idx3-ubyte");
    reader.ReadLabels("./MNIST/train-labels.idx1-ubyte");

    reader.ReadTrainingImages("./MNIST/t10k-images.idx3-ubyte");
    reader.ReadTrainingLabels("./MNIST/t10k-labels.idx1-ubyte");
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> loading_time = end - start;
    cout << "Total Loading time: " << loading_time.count() << " seconds" << endl; //Loading time: 4.1225 seconds

    reader.PrintImage(3502);
    cout << "Label of image 3502: ";
    reader.PrintLabel(3502);

    cout << endl;

    reader.PrintTrainingImage(9999);
    cout << "Label of training image 9999: ";
    reader.PrintTrainingLabel(9999);

    return 0;
}