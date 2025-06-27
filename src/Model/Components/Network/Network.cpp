#include "Network.h"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <functional>
#include <memory>

void Network::add(LayerType type,
                  int neuronCount,
                  Neuron::Activation::Type activation,
                  double dropoutRate,
                  bool isOutputLayer)
{
     switch (type)
     {
     case LayerType::DENSE:
     {
          if (neuronCount <= 0)
          {
               throw std::invalid_argument("Dense layer must have positive neuron count");
          }
          if (nextLayerInputSize == 0 && !layers.empty())
          {
               nextLayerInputSize = layers.back()->getOutputSize();
          }
          layers.push_back(std::make_unique<Dense>(
              neuronCount,
              nextLayerInputSize,
              activation,
              isOutputLayer));
          nextLayerInputSize = neuronCount;
          break;
     }
     case LayerType::DROPOUT:
     {
          if (dropoutRate <= 0.0 || dropoutRate >= 1.0)
          {
               throw std::invalid_argument("Dropout rate must be between 0 and 1");
          }
          if (layers.empty())
          {
               throw std::logic_error("Cannot add dropout layer as first layer");
          }
          layers.push_back(std::make_unique<Dropout>(dropoutRate));
          // Input size remains the same for next layer
          break;
     }
     default:
          throw std::invalid_argument("Unknown layer type");
     }
}

std::vector<double> Network::forward(const std::vector<double> &inputs)
{
     std::vector<double> result = inputs;

     for (auto &layer : layers)
     {
          result = layer->forward(result);
     }

     return result;
}

void Network::saveWeights(const std::string &filename)
{
     std::ofstream file(filename);
     if (!file.is_open())
     {
          throw std::runtime_error("Unable to open file to save weights.");
     }

     // Save network architecture first
     file << "Network Architecture:\n";
     for (const auto &layer : layers)
     {
          file << layer->getType() << " "
               << layer->getInputSize() << " "
               << layer->getOutputSize() << "\n";
     }
     file << "Weights:\n";

     // Save weights for trainable layers
     for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx)
     {
          if (layers[layerIdx]->isTrainableLayer())
          {
               layers[layerIdx]->save(file);
          }
     }
}

void Network::loadWeights(const std::string &filename)
{
     std::ifstream file(filename);
     if (!file)
     {
          throw std::runtime_error("Failed to open weights file: " + filename);
     }

     std::string line;
     bool readingWeights = false;
     size_t layerIdx = 0;

     while (std::getline(file, line))
     {
          if (line.empty())
               continue;

          if (line == "Weights:")
          {
               readingWeights = true;
               continue;
          }

          if (!readingWeights)
               continue; // Skip architecture info for now

          // Only load weights for trainable layers
          while (layerIdx < layers.size() && !layers[layerIdx]->isTrainableLayer())
          {
               layerIdx++;
          }

          if (layerIdx < layers.size())
          {
               layers[layerIdx]->load(file);
               layerIdx++;
          }
     }
}

void Network::backward(const std::vector<double> &expected, double learningRate)
{
     if (layers.empty())
          return;

     // Start with output gradient
     std::vector<double> gradient = expected;

     // Backward pass through layers in reverse order
     for (int i = layers.size() - 1; i >= 0; --i)
     {
          if (layers[i]->isTrainableLayer())
          {
               gradient = layers[i]->backward(gradient);
          }
          else
          {
               // For non-trainable layers like dropout, just pass through
               gradient = layers[i]->backward(gradient);
          }
     }

     // Update weights for trainable layers
     for (auto &layer : layers)
     {
          if (layer->isTrainableLayer())
          {
               layer->updateWeights(learningRate);
          }
     }
}

void Network::train(const std::vector<std::vector<double>> &inputs,
                    const std::vector<std::vector<double>> &targets,
                    int epochs, double learningRate, bool verbose,
                    std::function<void(float, float)> onEpochEnd)
{
     if (inputs.size() != targets.size())
     {
          throw std::invalid_argument("Number of input samples must match number of target samples.");
     }

     for (int epoch = 0; epoch < epochs; ++epoch)
     {
          double totalCost = 0.0;

          for (size_t i = 0; i < inputs.size(); ++i)
          {
               // Forward pass
               std::vector<double> prediction = forward(inputs[i]);

               // Compute loss
               for (size_t j = 0; j < prediction.size(); ++j)
               {
                    double error = prediction[j] - targets[i][j];
                    totalCost += error * error;
               }

               // Backward pass
               backward(targets[i], learningRate);
          }

          double averageCost = totalCost / inputs.size();
          if (verbose)
          {
               std::cout << "Epoch " << epoch + 1 << ", Cost: " << averageCost << "\n";
          }

          if (onEpochEnd)
          {
               onEpochEnd(epoch, float(averageCost));
          }
     }
}

void Network::setTrainingMode(bool training)
{
     for (auto &layer : layers)
     {
          if (auto dropout = dynamic_cast<Dropout *>(layer.get()))
          {
               dropout->setTraining(training);
          }
     }
}