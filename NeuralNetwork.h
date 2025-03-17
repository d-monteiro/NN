#include <stdio.h>
#include <vector>

using namespace std;

class NNLayer;
class NNNeuron;
class NNConnection;

typedef vector<NNLayer*>  VectorLayers;
typedef vector<double>  VectorWeights;
typedef vector<double>  VectorDeltas;
typedef vector<NNNeuron*>  VectorNeurons;
typedef vector<NNConnection> VectorConnections;

double Sigmoid(double x);

class NeuralNetwork{
public:
    NeuralNetwork(vector<int> nNeurons);
    virtual ~NeuralNetwork();

    void Initialize();

    void ForwardPropagate();

    void Train(vector<double>& input, vector<double>& target, double learningrate);

    // Helper functions
    void BackPropagate(vector<double>& target, double learningrate);
    double CalculateError(vector<double>& target);

    VectorLayers Layers;
    vector<double> Input;  // 784
    vector<double> Output; // 10
    double LearningRate;
};

class NNLayer{
public:
    NNLayer(int nNeurons);
    virtual ~NNLayer();

    void ForwardPropagate();

    // Helper functions
    void CalculateDeltas(vector<double>& target);
    void UpdateWeights(double learningrate);

    NNLayer* PreviousLayer;
    NNLayer* NextLayer;
    VectorNeurons Neurons;
    VectorDeltas Deltas;
};

class NNNeuron{
public:
    NNNeuron();
    virtual ~NNNeuron();

    //void AddConnection(); // Necessário??

    
    VectorWeights Weights;
    double Bias;
    double Output;

    // Helper variables
    double Delta;
    double InputSum;
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