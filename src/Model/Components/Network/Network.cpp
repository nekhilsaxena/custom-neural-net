#include "Network.h"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <functional>

Network::Network(const std::vector<int> &layerSizes, const std::vector<Neuron::Activation::Type> &activations)
{
     if (layerSizes.size() != activations.size())
          throw std::invalid_argument("Each layer must have an associated activation function.");

     // Create layers with specific activations
     for (size_t i = 1; i < layerSizes.size(); ++i)
     {
          bool isOutputLayer = (i == layerSizes.size() - 1);
          layers.emplace_back(layerSizes[i], layerSizes[i - 1], isOutputLayer, activations[i]); // assume addLayer exists
     }
}

std::vector<double> Network::forward(const std::vector<double> &inputs)
{
     std::vector<double> result = inputs;
     for (auto &layer : layers)
     {
          result = layer.forward(result);
     }
     return result;
}

void Network::saveWeights(const std::string &filename) const
{
     std::ofstream file(filename);
     if (!file.is_open())
          throw std::runtime_error("Unable to open file to save weights.");

     file << "Layer,Neuron";

     size_t maxWeights = layers.front().getNeuron(0).getWeights().size();
     for (size_t i = 1; i <= maxWeights; ++i)
          file << ",W" << i;
     file << ",Bias\n";

     for (size_t layerIdx = 0; layerIdx < layers.size(); ++layerIdx)
          layers[layerIdx].saveWeights(file, layerIdx);
}

void Network::loadWeights(const std::string &filename)
{
     std::ifstream file(filename);
     if (!file)
          throw std::runtime_error("Failed to open weights file: " + filename);

     std::string line;

     // Skip header if present
     if (std::getline(file, line))
     {
          if (line.rfind("Layer", 0) != 0)
               file.seekg(0);
     }

     layers.clear();

     while (std::getline(file, line))
     {
          std::stringstream ss(line);
          std::string value;

          std::getline(ss, value, ',');
          int layerIndex = std::stoi(value);
          std::getline(ss, value, ',');
          int neuronIndex = std::stoi(value);

          std::vector<double> numbers;
          while (std::getline(ss, value, ','))
               numbers.push_back(std::stod(value));

          if (numbers.empty())
               throw std::runtime_error("No weights in row.");

          std::vector<double> weights(numbers.begin(), numbers.end() - 1);
          double bias = numbers.back();
          size_t numInputs = weights.size();

          // Create layers if they don’t exist
          while (layers.size() <= static_cast<size_t>(layerIndex))
          {
               size_t inputSize = (layerIndex == 0)
                                      ? numInputs
                                      : layers[layerIndex - 1].size();

               layers.emplace_back(0, int(inputSize), false, Neuron::Activation::SIGMOID);
          }

          Layer &layer = layers[layerIndex];

          // Add neurons if needed
          while (layer.size() <= static_cast<size_t>(neuronIndex))
          {
               layer.addNeuron(Neuron(numInputs, false)); // isOutput will be set later
          }

          Neuron &neuron = layer.getNeuron(neuronIndex);
          neuron.setAll(weights, bias);

          std::cout << "Layer " << layerIndex << ", Neuron " << neuronIndex << ":\n";
          std::cout << "  Weights: ";
          for (double w : weights)
               std::cout << w << " ";
          std::cout << "\n  Bias: " << bias << "\n";
     }

     if (!layers.empty())
     {
          Layer &outputLayer = layers.back();
          for (Neuron &n : outputLayer.getNeurons())
          {
               n.isOutputNeuron = true;
          }
     }

     std::cout << "\n  ****Weights Have Been Loaded!****" << "\n";
}

void Network::backward(const std::vector<double> &expected, double learningRate)
{
     if (layers.empty())
          return;

     // --- 1. Output layer error ---
     Layer &outputLayer = layers.back();
     for (size_t i = 0; i < outputLayer.size(); ++i)
     {
          Neuron &n = outputLayer.getNeuron(i);
          double error = n.output - expected[i];

          // Derivative depends on activation:
          n.delta = error * (n.isOutputNeuron ? n.sigmoidDerivative(n.output) : 1.0);
     }

     // --- 2. Hidden layers (backward pass) ---
     for (int l = layers.size() - 2; l >= 0; --l)
     { // From second-last to first
          Layer &currentLayer = layers[l];
          Layer &nextLayer = layers[l + 1];

          for (size_t i = 0; i < currentLayer.size(); ++i)
          {
               Neuron &n = currentLayer.getNeuron(i);
               double sum = 0.0;

               // Sum weighted errors from next layer
               for (size_t j = 0; j < nextLayer.size(); ++j)
               {
                    sum += nextLayer.getNeuron(j).weights[i] * nextLayer.getNeuron(j).delta;
               }

               // ReLU derivative for hidden layers
               n.delta = sum * n.reluDerivative(n.output);
          }
     }

     // --- 3. Update weights and biases ---
     for (size_t l = 0; l < layers.size(); ++l)
     {
          Layer &layer = layers[l];
          const std::vector<double> &inputs = (l == 0) ? layer.lastInput : layers[l - 1].getOutputs();

          for (Neuron &n : layer.getNeurons())
          {
               for (size_t w = 0; w < n.weights.size(); ++w)
               {
                    n.weights[w] -= learningRate * n.delta * inputs[w];
               }
               n.bias = n.getBias() - learningRate * n.delta;
          }
     }
}

void Network::train(const std::vector<std::vector<double>> &inputs,
                    const std::vector<std::vector<double>> &targets,
                    int epochs, double learningRate, bool verbose,
                    std::function<void(float, float)> onEpochEnd)
{
     if (inputs.size() != targets.size())
          throw std::invalid_argument("Number of input samples must match number of target samples.");

     for (int epoch = 0; epoch < epochs; ++epoch)
     {
          double totalCost = 0.0;

          for (size_t i = 0; i < inputs.size(); ++i)
          {
               // 1. Forward pass
               std::vector<double> prediction = forward(inputs[i]);

               // 2. Compute loss
               for (size_t j = 0; j < prediction.size(); ++j)
               {
                    double error = prediction[j] - targets[i][j];
                    totalCost += error * error;
               }

               // 3. Backward pass
               backward(targets[i], learningRate);
          }

          double averageCost = totalCost / inputs.size();
          if (verbose)
               std::cout << "Epoch " << epoch + 1 << ", Cost: " << averageCost << "\n";

          if (onEpochEnd)
               onEpochEnd(epoch, float(averageCost));
     }
}