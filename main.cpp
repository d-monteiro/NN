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
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> loading_time = end - start;
    cout << "Loading time: " << loading_time.count() << " seconds" << endl; //Loading time: 2.1043 seconds

    reader.PrintImage(0);

    return 0;
}