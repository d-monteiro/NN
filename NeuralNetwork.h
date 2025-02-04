#include <stdio.h>
#include <vector>

using namespace std;

typedef vector< NNLayer* >  VectorLayers;
typedef vector< NNWeight* >  VectorWeights;
typedef vector< NNNeuron* >  VectorNeurons;
typedef vector< NNConnection > VectorConnections;

class NeuralNetwork{
public:
    NeuralNetwork();
    virtual ~NeuralNetwork();

    void ForwardPropagate();

    void BackPropagate();
};

class NNLayer{
public:
    NNLayer();
    virtual ~NNLayer();

    void ForwardPropagate();

    void BackPropagate();
};

class NNNeuron{
public:
    NNNeuron();
    virtual ~NNNeuron();

    void AddConnection();
};

class NNConnection{
public:
    NNConnection();
    virtual ~NNConnection();
};

class NNWeight{
public:
    NNWeight();
    virtual ~NNWeight();
};