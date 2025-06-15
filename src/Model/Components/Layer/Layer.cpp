#include "Layer.h"
#include <cstdlib>
#include <fstream>

Layer::Layer(int numNeurons, int numInputsPerNeuron, bool isOutputLayer, Neuron::Activation::Type activation)
{
     for (int i = 0; i < numNeurons; ++i)
     {
          neurons.emplace_back(numInputsPerNeuron, isOutputLayer, activation);
     }
}

std::vector<double> Layer::forward(const std::vector<double> &inputs)
{
     lastInput = inputs;

     std::vector<double> outputs;
     for (auto &neuron : neurons)
     {
          outputs.push_back(neuron.forward(inputs));
     }
     return outputs;
}

void Layer::saveWeights(std::ofstream &file, size_t layerIndex) const
{
     for (size_t i = 0; i < neurons.size(); ++i)
     {
          neurons[i].saveWeights(file, layerIndex, i);
     }
}

std::vector<Neuron> &Layer::getNeurons()
{
     return neurons;
}

const std::vector<Neuron> &Layer::getNeurons() const
{
     return neurons;
}

Neuron &Layer::getNeuron(int index)
{
     return neurons.at(index);
}

const Neuron &Layer::getNeuron(int index) const
{
     return neurons.at(index);
}

void Layer::resizeLayer(int neuronIndex, Neuron neuron)
{
     neurons.resize(neuronIndex + 1, neuron);
}

size_t Layer::size() const
{
     return neurons.size();
}

void Layer::addNeuron(const Neuron &neuron)
{
     neurons.push_back(neuron);
}

std::vector<double> Layer::getOutputs() const
{
     std::vector<double> outputs;
     for (const Neuron &n : neurons)
          outputs.push_back(n.output);
     return outputs;
}

// Compute deltas for the output layer
void Layer::computeDeltas(const std::vector<double> &targets)
{
     for (size_t i = 0; i < neurons.size(); ++i)
     {
          neurons[i].computeOutputDelta(targets[i]);
     }
}

// Compute deltas for a hidden layer
void Layer::computeDeltas(const Layer &nextLayer)
{
     for (size_t i = 0; i < neurons.size(); ++i)
     {
          neurons[i].computeHiddenDelta(nextLayer.getNeurons(), static_cast<int>(i));
     }
}

// Update the weights of each neuron in this layer
void Layer::updateWeights(const std::vector<double> &inputs, double learningRate)
{
     for (auto &neuron : neurons)
     {
          neuron.updateWeights(inputs, learningRate);
     }
}
