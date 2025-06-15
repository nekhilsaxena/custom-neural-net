#ifndef LAYER_H
#define LAYER_H

#include "../Neuron/Neuron.h"
#include <vector>
#include <cstdlib>
#include <fstream>

class Layer
{
private:
     std::vector<Neuron> neurons;

public:
     // Constructor to initialize neurons
     Layer(int numNeurons, int numInputsPerNeuron, bool isOutputLayer, Neuron::Activation::Type activation);

     // Forward pass
     std::vector<double> forward(const std::vector<double> &inputs);

     // Save weights of neurons to file
     void saveWeights(std::ofstream &file, size_t layerIndex) const;

     // Getters for neurons
     std::vector<Neuron> &getNeurons();
     const std::vector<Neuron> &getNeurons() const;

     // Get a specific neuron
     Neuron &getNeuron(int index);
     const Neuron &getNeuron(int index) const;

     // Resize the layer by adding neurons
     void resizeLayer(int neuronIndex, Neuron neuron);
     size_t size() const;
     void addNeuron(const Neuron &neuron);

     // Get the outputs of all neurons in the layer
     std::vector<double> getOutputs() const;

     // Backpropagation methods
     std::vector<double> lastInput; // Last inputs to this layer

     // Compute deltas for output layer neurons
     void computeDeltas(const std::vector<double> &targets);

     // Compute deltas for hidden layer neurons
     void computeDeltas(const Layer &nextLayer);

     // Update weights of all neurons in the layer
     void updateWeights(const std::vector<double> &inputs, double learningRate);
};

#endif
