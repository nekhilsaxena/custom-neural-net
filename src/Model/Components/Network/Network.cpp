#include "Network.h"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <functional>
#include <memory>

void Network::initialize(int inputSize)
{
     input_dim = size_t(inputSize);
}

void Network::add(LayerType type,
                  int neuronCount,
                  Neuron::Activation::Type activation,
                  double dropoutRate)
{
     int layerInputs = layers.empty() ? input_dim : layers.back()->getOutputSize();

     std::cout << "Layer Inputs: " << layerInputs << "\n";

     switch (type)
     {
     case LayerType::DENSE:
     {
          if (neuronCount <= 0)
          {
               throw std::invalid_argument("Dense layer must have positive neuron count");
          }

          layers.push_back(std::make_unique<Dense>(
              neuronCount,
              layerInputs,
              activation));
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
          layers.push_back(std::make_unique<Dropout>(dropoutRate, layerInputs));
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

void Network::backward(const std::vector<double> &expected, double learningRate)
{
     if (layers.empty())
          return;

     // Start with output gradient
     std::vector<double> gradient = expected;

     // Prepare a vector to store next layer weights for each backward pass
     std::vector<std::vector<double>> nextLayerWeights;

     // Backward pass through layers in reverse order
     for (int i = layers.size() - 1; i >= 0; --i)
     {
          if (layers[i]->isTrainableLayer())
          {
               // For trainable layers (like Dense)
               if (i == layers.size() - 1)
               {
                    // Output layer case - no next layer weights needed
                    gradient = layers[i]->backward(gradient);
               }
               else
               {
                    // Hidden layer case - pass next layer's weights
                    auto *nextLayer = dynamic_cast<Dense *>(layers[i + 1].get());
                    if (nextLayer)
                    {
                         nextLayerWeights = nextLayer->getWeightsMatrix();
                    }
                    gradient = layers[i]->backward(gradient, nextLayerWeights);
               }
          }
          else
          {
               // For non-trainable layers (like Dropout)
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
     // 1. Input validation
     if (inputs.empty() || targets.empty())
     {
          throw std::invalid_argument("Inputs and targets cannot be empty");
     }
     if (inputs.size() != targets.size())
     {
          throw std::invalid_argument("Inputs and targets size mismatch");
     }

     // 2. Input normalization (per feature)
     std::vector<double> means, stddevs;
     auto normalizedInputs = normalizeInputs(inputs, means, stddevs);

     // Store normalization parameters for later use (e.g., in predict)
     inputMeans = means;
     inputStddevs = stddevs;

     // 3. Training loop with learning rate decay
     for (int epoch = 0; epoch < epochs; ++epoch)
     {
          double totalCost = 0.0;
          double currentLearningRate = learningRate * (1.0 / (1.0 + 0.01 * epoch)); // Simple decay
          std::vector<double> prediction;
          for (size_t i = 0; i < normalizedInputs.size(); ++i)
          {
               // Forward pass
               prediction = forward(normalizedInputs[i]);

               // Compute loss
               double sampleCost = 0.0;
               for (size_t j = 0; j < prediction.size(); ++j)
               {
                    double error = prediction[j] - targets[i][j];
                    sampleCost += error * error;
               }
               totalCost += sampleCost;

               // Backward pass
               backward(targets[i], currentLearningRate);
          }

          // Reporting
          double averageCost = totalCost / normalizedInputs.size();
          if (verbose)
          {
               std::cout << "Epoch " << epoch + 1
                         << ", Cost: " << averageCost
                         << ", LR: " << currentLearningRate
                         << ", Final Prediction: " << (prediction.empty() ? 0.0 : prediction[0])
                         << "\n";
          }

          if (onEpochEnd)
          {
               onEpochEnd(epoch, float(averageCost));
          }

          // Optional early stopping
          if (averageCost < 0.001)
          {
               if (verbose)
                    std::cout << "Early stopping at epoch " << epoch + 1 << "\n";
               break;
          }
     }
}

std::vector<double> Network::predict(std::vector<double> input)
{
     std::vector<double> normalizedInput = this->normalizeInput(input);
     std::vector<double> output = this->forward(normalizedInput);

     return output;
}