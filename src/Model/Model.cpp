#include "Model.h"
#include <stdexcept>
#include <iostream>

Model::Model()
{
     // First layer is added implicitly when build() is called with inputs
}

void Model::add(int neurons, Neuron::Activation::Type activation)
{
     layerSizes.push_back(neurons);
     activations.push_back(activation);
}

void Model::build()
{
     if (layerSizes.empty())
          throw std::runtime_error("No layers added to the model.");

     network = std::make_unique<Network>(layerSizes, activations);
}

void Model::train(const std::vector<std::vector<double>> &inputs, const std::vector<double> &outputs, int epochs, double learningRate)
{
     if (!network)
          throw std::runtime_error("Model must be built before training.");

     std::vector<std::vector<double>> targets;
     for (double val : outputs)
          targets.push_back({val}); // Convert to vector<double> for network compatibility

     network->train(inputs, targets, epochs, learningRate, true);
}

std::vector<double> Model::predict(const std::vector<double> &input)
{
     if (!network)
          throw std::runtime_error("Model must be built before prediction.");

     return network->forward(input);
}
