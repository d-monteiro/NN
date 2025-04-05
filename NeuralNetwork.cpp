#include <stdio.h>
#include <vector>
#include <random>
#include <cmath>
#include "NeuralNetwork.h"

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(-0.5, 0.5);

double Tanh(double x) {
    return tanh(x);
}

double DTanh(double x) {
    return 1.0 - tanh(x) * tanh(x);
}

/*------------NEURAL NETWORK------------*/

NeuralNetwork::NeuralNetwork(vector<int> nNeurons){
    if(nNeurons.size() <= 1) {
        printf("Number of layers must be greater than 1.\n");
        return;
    }

    // Criação
    for(int i = 0; i < nNeurons.size(); ++i){
        NNLayer* layer = new NNLayer(nNeurons[i]);
        Layers.push_back(layer);
    }
    
    // Ligação
    for (size_t i = 0; i < Layers.size(); ++i) {
        if (i > 0) Layers[i]->PreviousLayer = Layers[i-1];
        if (i < Layers.size() - 1) Layers[i]->NextLayer = Layers[i+1];
    }
    
    // Weights
    for(size_t i = 0; i < Layers.size() - 1; ++i){
        for (NNNeuron* neuron : Layers[i]->Neurons){
            neuron->Weights.resize(Layers[i+1]->Neurons.size());
        }
    }
}

NeuralNetwork::~NeuralNetwork(){
    for(NNLayer* layer : Layers){
        delete layer;
    }
}

void NeuralNetwork::Initialize(){
    for(size_t i = 0; i < Layers.size() - 1; ++i){
        for(NNNeuron* neuron : Layers[i]->Neurons){
            neuron->Bias = dis(gen);

            for(size_t j = 0; j < neuron->Weights.size(); ++j){
                neuron->Weights[j] = dis(gen);
            }
        }
    }
}

void NeuralNetwork::ForwardPropagate(){
    VectorLayers::iterator Layers_it = Layers.begin();

    for(int i = 0; i < Input.size(); ++i){
        (*Layers_it)->Neurons[i]->Output = Input[i];
    }

    Layers_it++;

    while(Layers_it != Layers.end()){
        (*Layers_it)->ForwardPropagate();
        Layers_it++;
    }
}

void NeuralNetwork::Train(vector<double>& input, vector<double>& target, double learningrate){
    // 0. Set input
    Input = input;

    // 1. Calculate the output
    ForwardPropagate();
    
    // 2. Calculate the error
    double error = CalculateError(target);
    
    // 3. Backpropagate the error
    BackPropagate(target, learningrate);
}

double NeuralNetwork::CalculateError(vector<double>& target){
    // Mean Squared Error calculation
    double error = 0.0;
    NNLayer* outputLayer = Layers.back();
    
    for (size_t i = 0; i < outputLayer->Neurons.size(); ++i) {
        double diff = target[i] - outputLayer->Neurons[i]->Output;
        error += diff * diff;
    }
    
    return error / outputLayer->Neurons.size();
}

void NeuralNetwork::BackPropagate(vector<double>& target, double learningrate){
    // 1. Calculate layer deltas (working backwards from the output)
    for (int i = Layers.size() - 2; i > 0; --i) {
        Layers[i]->CalculateDeltas(target);
    }
    
    // 2. Update all weights
    for (int i = Layers.size() - 1; i > 0; --i) {
        Layers[i]->UpdateWeights(learningrate);
    }
}

/*------------LAYER------------*/

NNLayer::NNLayer(int nNeurons){
    if(nNeurons <= 0){
        printf("Number of neurons per layer must be greater than 0.\n");
        return;
    }

    for (int i = 0; i < nNeurons; ++i){
        NNNeuron* neuron = new NNNeuron();
        Neurons.push_back(neuron);
    }

    //Espero que não dê asneira
    //Weights.resize(nNeurons);
    //Biases.resize(nNeurons);
}

NNLayer::~NNLayer(){
    for (NNNeuron* neuron : Neurons){
        delete neuron;
    }
}

void NNLayer::ForwardPropagate(){
    if (PreviousLayer == nullptr){
        printf("Previous layer is null.\n");
        return;
    }

    for (size_t i = 0; i < Neurons.size(); ++i){
        double sum = 0.0;

        for (size_t j = 0; j < PreviousLayer->Neurons.size(); ++j){
            sum += PreviousLayer->Neurons[j]->Output * PreviousLayer->Neurons[j]->Weights[i];
        }

        sum += Neurons[i]->Bias;

        Neurons[i]->InputSum = sum;
        Neurons[i]->Output = Tanh(sum);
    }
}

void NNLayer::CalculateDeltas(vector<double>& target){
    Deltas.resize(Neurons.size());
    
    // Error delta = (target - output) * DTanh
    
    // Para a última layer só 
    if(NextLayer == nullptr){
        for(size_t i = 0; i < Neurons.size(); ++i){
            double output = Neurons[i]->Output;
            Neurons[i]->Delta = (target[i] - output) * DTanh(Neurons[i]->InputSum);
            Deltas[i] = Neurons[i]->Delta;
        }
    }
    
    // Somar erros da layer acima
    for(size_t i = 0; i < Neurons.size(); ++i){
        double error = 0.0; // Aqui? Acho que sim!

        for(size_t j = 0; j < NextLayer->Neurons.size(); ++j){
            // Corrigir com novo refactor
            error += NextLayer->Neurons[j]->Delta * Neurons[i]->Weights[j];
        }

        Neurons[i]->Delta = error * DTanh(Neurons[i]->InputSum);
        Deltas[i] = Neurons[i]->Delta;
    }
}
    
void NNLayer::UpdateWeights(double learningrate){
    if (PreviousLayer == nullptr) return;

    // Weight_ij += learning_rate * delta_i * output_j
    
    // Um neuron de cada vez
    for (size_t i = 0; i < Neurons.size(); ++i){
        for (size_t j = 0; j < PreviousLayer->Neurons.size(); ++j){
            PreviousLayer->Neurons[j]->Weights[i] += learningrate * Neurons[i]->Delta * PreviousLayer->Neurons[j]->Output;
        }
    }
}

/*------------NEURON------------*/

NNNeuron::NNNeuron(){

}

NNNeuron::~NNNeuron(){
    Weights.clear();
}

/*------------CONNECTION------------

NNConnection::NNConnection(){ // Will this be needed?

}

NNConnection::~NNConnection(){

}

/*------------WEIGHT------------

NNWeight::NNWeight(){ // Will this be needed?

}

NNWeight::~NNWeight(){

}
*/