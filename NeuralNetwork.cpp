#include <stdio.h>
#include <vector>
#include <random>
#include "NeuralNetwork.h"

using namespace std;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<> dis(-0.5, 0.5);

double Sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

/*------------NEURAL NETWORK------------*/

NeuralNetwork::NeuralNetwork(vector<int> nNeurons){

    if(nNeurons.size() <= 1){
        printf("Number of layers must be greater than 1.\n");
        return;
    }

    for (int i = 0; i < nNeurons.size(); ++i){
        NNLayer* layer = new NNLayer(nNeurons[i]);

        if (i > 0) layer->PreviousLayer = Layers.back();

        Layers.push_back(layer);
    }
}

NeuralNetwork::~NeuralNetwork(){
    for (NNLayer* layer : Layers){
        delete layer;
    }
}

void NeuralNetwork::Initialize(){
    for(NNLayer* layer : Layers){
        for(double& weight : layer->Weights){
            weight = dis(gen);
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

void NeuralNetwork::BackPropagate(){

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

    Weights.resize(nNeurons);
}

NNLayer::~NNLayer(){
    for (NNNeuron* neuron : Neurons){
        delete neuron;
    }
}

void NNLayer::ForwardPropagate() {
    if (PreviousLayer == nullptr){
        printf("Previous layer is null.\n");
        return;
    }

    for (size_t i = 0; i < Neurons.size(); ++i){
        double sum = 0.0;
        for (size_t j = 0; j < PreviousLayer->Neurons.size(); ++j){
            sum += PreviousLayer->Neurons[j]->Output * Weights[i];
        }
        Neurons[i]->Output = Sigmoid(sum);
    }
}

void NNLayer::BackPropagate(){

}

/*------------NEURON------------*/

NNNeuron::NNNeuron(){

}

NNNeuron::~NNNeuron(){

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