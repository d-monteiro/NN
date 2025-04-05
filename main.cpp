#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <algorithm>

#include "MNIST.h"
#include "NeuralNetwork.h"

using namespace std;

//g++ main.cpp MNIST.cpp NeuralNetwork.cpp -o main && ./main

//Helper function
int getPrediction(const vector<NNNeuron*>& outputNeurons) {
    int maxIndex = 0;
    double maxValue = outputNeurons[0]->Output;
    
    for (int i = 1; i < outputNeurons.size(); i++) {
        if (outputNeurons[i]->Output > maxValue) {
            maxValue = outputNeurons[i]->Output;
            maxIndex = i;
        }
    }
    
    return maxIndex;
}

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
    cout << endl;
    
    auto start2 = chrono::high_resolution_clock::now();
    nn.Initialize();
    auto end2 = chrono::high_resolution_clock::now();
    chrono::duration<double> loading_time2 = end2 - start2;
    
    cout << "Creation time: " << loading_time1.count() << " seconds" << endl;
    cout << "Initialization time: " << loading_time2.count() << " seconds" << endl;
    cout << "Total time: " << loading_time1.count() + loading_time2.count() << endl;
    cout << endl;
     
    // PART 3: Training
    int epochs = 3;
    double learningRate = 0.1;
    int batchSize = 32  ;
    int TrainingSamples = 60000;
    int numTestSamples = 10000;

    cout << epochs << " epochs" << ", Learning rate: " << learningRate << endl;
    
    for(int epoch = 0; epoch < epochs; epoch++){
        auto epochStart = chrono::high_resolution_clock::now();
        double totalError = 0.0;
        int correctPredictions = 0;
        
        // Shuffle indices
        vector<int> indices(TrainingSamples);
        for(int i = 0; i < TrainingSamples; i++){
            indices[i] = i;
        }
        random_device rd;
        mt19937 g(rd());
        shuffle(indices.begin(), indices.end(), g);
        
        // Epochs
        for(int batch = 0; batch < TrainingSamples / batchSize; batch++){
            double batchError = 0.0;
            int batchCorrect = 0;
            
            for(int i = 0; i < batchSize; i++){
                int idx = indices[batch * batchSize + i];
                
                vector<double> input = reader.imageToInput(reader.Images[idx]);
                vector<double> target = reader.labelToTarget(reader.Labels[idx]);
                
                // Train on example
                nn.Train(input, target, learningRate);
                
                batchError += nn.CalculateError(target);
                
                // Check if prediction is correct
                int prediction = getPrediction(nn.Layers.back()->Neurons);
                if(prediction == reader.Labels[idx]){
                    batchCorrect++;
                }
            }
            
            totalError += batchError / batchSize;
            correctPredictions += batchCorrect;
            
            // Print progress every 10 batches
            if(batch % 10 == 0){
                cout << "\rEpoch " << epoch + 1 << "/" << epochs 
                     << " - Batch " << batch << "/" << TrainingSamples / batchSize
                     << " - Error: " << fixed << setprecision(4) << batchError / batchSize 
                     << " - Accuracy: " << (100.0 * batchCorrect / batchSize) << "%";
                cout.flush();
            }
        }
        
        auto epochEnd = chrono::high_resolution_clock::now();
        chrono::duration<double> epochTime = epochEnd - epochStart;
        
        // Calculate epoch metrics
        double avgError = totalError / (TrainingSamples / batchSize);
        double accuracy = 100.0 * correctPredictions / TrainingSamples;
        
        cout << "\nEpoch " << epoch + 1 << " completed in " << epochTime.count() 
             << " seconds. Avg Error: " << avgError
             << " - Training Accuracy: " << accuracy << "%" << endl;
        
        /*
        // Evaluate on test set after each epoch
        if (numTestSamples > 0) {
            int testCorrect = 0;
            double testError = 0.0;
            
            cout << "Evaluating on test set..." << endl;
            
            for (int i = 0; i < numTestSamples; i++) {
                vector<double> testInput = reader.imageToInput(reader.TrainingImages[i]);
                vector<double> testTarget = reader.labelToTarget(reader.TrainingLabels[i]);
                
                // Forward pass only (no training)
                nn.Input = testInput;
                nn.ForwardPropagate();
                
                testError += nn.CalculateError(testTarget);
                
                int prediction = getPrediction(nn.Layers.back()->Neurons);
                if (prediction == reader.TrainingLabels[i]) {
                    testCorrect++;
                }
                
                // Print progress
                if (i % 1000 == 0) {
                    cout << "\rTesting: " << i << "/" << numTestSamples;
                    cout.flush();
                }
            }
            
            double testAccuracy = 100.0 * testCorrect / numTestSamples;
            cout << "\nTest Error: " << testError / numTestSamples 
                 << " - Test Accuracy: " << testAccuracy << "%" << endl;
        }
        */

        // Ver se vale a pena
        if (epoch > 2) {
            learningRate *= 0.8;
            cout << "Reduced learning rate to " << learningRate << endl;
        }
    }




    // PART 4: Forward Propagation
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

    return 0;
}