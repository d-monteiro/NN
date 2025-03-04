#include <stdio.h>
#include <vector>

using namespace std;

class NNLayer;
class NNNeuron;
class NNConnection;

typedef vector<NNLayer*>  VectorLayers;
typedef vector<double>  VectorWeights;
typedef vector<NNNeuron*>  VectorNeurons;
typedef vector<NNConnection> VectorConnections;

double Sigmoid(double x);

class NeuralNetwork{
public:
    NeuralNetwork(vector<int> nNeurons);
    virtual ~NeuralNetwork();

    void Initialize();

    void ForwardPropagate();

    void BackPropagate();

    VectorLayers Layers;
    vector<double> Input;
    vector<double> Output;
};

class NNLayer{
public:
    NNLayer(int nNeurons);
    virtual ~NNLayer();

    void ForwardPropagate();

    void BackPropagate();

    NNLayer* PreviousLayer;
    VectorNeurons Neurons;
    VectorWeights Weights;
};

class NNNeuron{
public:
    NNNeuron();
    virtual ~NNNeuron();

    void AddConnection();

    double Output;
};

/*
class NNConnection{ // Will this be needed?
public:
    NNConnection();
    virtual ~NNConnection();
};

class NNWeight{ // Will this be needed?
public:
    NNWeight();
    virtual ~NNWeight();

    double Value; // Or float?
};
*/