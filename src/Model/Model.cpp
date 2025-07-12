#include "Model.h"
#include <iostream>
#include <fstream>

void Model::add(int units, Neuron::Activation::Type activation)
{
     if (built)
     {
          throw std::runtime_error("Cannot add layers after model is built");
     }
     if (units <= 0)
     {
          throw std::invalid_argument("Number of units must be positive");
     }

     // Add main dense layer
     layers.push_back({
         Network::LayerType::DENSE,
         units,
         activation,
         0.0 // No dropout for the dense layer itself
     });
}

void Model::add(double dropoutRate)
{
     if (dropoutRate < 0 || dropoutRate >= 1)
     {
          throw std::invalid_argument("Dropout rate must be in [0, 1)");
     }

     // Add dropout layer if specified (except for output layer)
     if (dropoutRate > 0)
     {
          layers.push_back({Network::LayerType::DROPOUT,
                            0,                          // Not used for dropout
                            Neuron::Activation::LINEAR, // Not used for dropout
                            dropoutRate});
     }
}

void Model::build(int inputSize)
{
     if (built)
     {
          throw std::runtime_error("Model already built");
     }
     if (inputSize <= 0)
     {
          throw std::invalid_argument("Input size must be positive");
     }
     if (layers.empty())
     {
          throw std::runtime_error("No layers added to model");
     }

     network = std::make_unique<Network>();

     size_t currentInputSize = size_t(inputSize);

     network->initialize(inputSize);

     // Create all layers in the network
     for (size_t i = 0; i < layers.size(); i++)
     {
          const auto &config = layers[i];

          switch (config.type)
          {
          case Network::LayerType::DENSE:
               std::cout << "Layer Size: " << config.neurons << "\n";
               network->add(
                   config.type,
                   config.neurons,
                   config.activation,
                   0.0); // Not used
               currentInputSize = config.neurons;
               break;

          case Network::LayerType::DROPOUT:
               network->add(
                   config.type,
                   0,                          // Not used
                   Neuron::Activation::LINEAR, // Not used
                   config.dropoutRate);
               break;
          }
     }

     built = true;
     std::cout << "Model built with input size " << inputSize
               << " and output size " << network->getOutputSize() << "\n";
}

void Model::train(const std::vector<std::vector<double>> &inputs,
                  const std::vector<double> &outputs,
                  int epochs,
                  double learningRate)
{
     if (!built)
     {
          throw std::runtime_error("Model must be built before training");
     }
     if (inputs.empty() || outputs.empty())
     {
          throw std::invalid_argument("Training data cannot be empty");
     }
     if (inputs.size() != outputs.size())
     {
          throw std::invalid_argument("Inputs and outputs must have same length");
     }

     // Convert outputs to 2D vector
     std::vector<std::vector<double>> targets;
     for (double val : outputs)
     {
          targets.push_back({val});
     }

     network->setTrainingMode(true);
     network->train(inputs, targets, epochs, learningRate, true);
     network->setTrainingMode(false);
}

std::vector<double> Model::predict(const std::vector<double> &input)
{
     if (!built)
     {
          throw std::runtime_error("Model must be built before prediction");
     }
     if (input.size() != network->getInputSize())
     {
          std::cout << "Input Size Given: " << input.size() << "\n";
          std::cout << "Input Size Expected: " << network->getInputSize() << "\n";
          throw std::invalid_argument("Input size mismatch");
     }

     network->setTrainingMode(false);
     return network->predict(input);
}

void Model::save(const std::string &filename)
{
     if (!built)
     {
          throw std::runtime_error("Model must be built before saving");
     }
     network->saveWeights(filename);
     std::cout << "Model saved to " << filename << "\n";
}

void Model::load(const std::string &filename)
{
     if (built)
     {
          throw std::runtime_error("Model already built, cannot load weights");
     }

     network = std::make_unique<Network>(); // Create a new empty network

     network->loadWeights(filename); // This will fully create and initialize all layers

     built = true;

     std::cout << "Model loaded from " << filename << "\n";
}
