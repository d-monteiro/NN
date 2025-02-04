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
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> loading_time = end - start;
    cout << "Loading time: " << loading_time.count() << " seconds" << endl; //Loading time: 4.1225 seconds

    reader.PrintImage(3502);
    cout << "Label of image 3502: ";
    reader.PrintLabel(3502);

    return 0;
}