#include "Dense.h"
#include "../../Neuron/Neuron.h"
#include "../../Layer/Layer.h"
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <iostream>
#include <algorithm>

Dense::Dense(int numNeurons, int numInputsPerNeuron,
             Neuron::Activation::Type activation)
    : Layer(), // Base constructor
      lastDeltas()
{
     isTrainable = true; // Set base class protected member

     numInputs = numInputsPerNeuron;

     if (numNeurons <= 0 || numInputsPerNeuron <= 0)
     {
          std::cout << numNeurons << "\n";
          std::cout << numInputsPerNeuron << "\n";
          throw std::invalid_argument("Number of neurons and inputs must be positive");
     }

     neurons.reserve(numNeurons);
     for (int i = 0; i < numNeurons; ++i)
     {
          neurons.emplace_back(numInputsPerNeuron, activation);
     }
}

std::vector<double> Dense::forward(const std::vector<double> &inputs)
{
     if (inputs.size() != getInputSize())
     {
          std::cout << inputs.size() << "\n";
          std::cout << getInputSize() << "\n";
          throw std::invalid_argument("Input size mismatch in Dense forward pass");
     }

     lastInput = inputs;
     lastOutput.resize(neurons.size());

     for (size_t i = 0; i < neurons.size(); ++i)
     {
          lastOutput[i] = neurons[i].forward(inputs);
     }

     return lastOutput;
}

std::vector<double> Dense::backward(const std::vector<double> &gradients,
                                    const std::vector<std::vector<double>> &nextLayerWeights)
{
     // 1. Input validation
     if (gradients.empty())
          throw std::runtime_error("Empty gradients");
     if (gradients.size() != neurons.size())
     {
          throw std::invalid_argument("Gradient/neuron count mismatch");
     }

     // 2. Initialize with numerical stability safeguards
     lastDeltas.resize(neurons.size());
     std::vector<double> outputGradients(getInputSize(), 0.0);
     const double epsilon = 1e-8; // Safety margin
     bool isOutputLayer = nextLayerWeights.empty();

     // 3. Calculate deltas with gradient preservation
     for (size_t i = 0; i < neurons.size(); ++i)
     {
          Neuron &n = neurons[i];
          double delta;

          if (isOutputLayer)
          {
               // Output layer - direct error
               double error = n.output - gradients[i];
               delta = error * activationDerivativeSafe(n);
          }
          else
          {
               // Hidden layer - weighted sum from next layer
               double sum = 0.0;
               for (size_t j = 0; j < nextLayerWeights.size(); ++j)
               {
                    if (i >= nextLayerWeights[j].size())
                         continue;
                    sum += nextLayerWeights[j][i] * gradients[j];
               }
               delta = sum * activationDerivativeSafe(n);
          }

          // Normalize instead of clip
          double norm_factor = std::max(1.0, fabs(delta) / 5.0); // Adjust denominator as needed
          n.delta = delta / norm_factor;
          lastDeltas[i] = n.delta;

          // Check for dead neurons
          // if (fabs(n.delta) < 1e-7)
          // {
          //      std::cerr << "Warning: Near-zero delta for neuron " << i << "\n";
          // }
     }

     // 4. Gradient propagation with stability checks
     for (size_t i = 0; i < getInputSize(); ++i)
     {
          for (size_t j = 0; j < neurons.size(); ++j)
          {
               const auto &weights = neurons[j].getWeights();
               if (i >= weights.size())
                    continue;

               double contribution = weights[i] * lastDeltas[j];

               // Numerical stability guard
               if (std::isinf(contribution))
               {
                    contribution = (contribution > 0) ? 1e8 : -1e8;
               }
               else if (std::isnan(contribution))
               {
                    contribution = 0.0;
               }

               outputGradients[i] += contribution;
          }
     }

     return outputGradients;
}

// Safe derivative calculation
double Dense::activationDerivativeSafe(const Neuron &n) const
{
     double deriv;
     switch (n.getActivation())
     {
     case Neuron::Activation::RELU:
          deriv = (n.output > 0) ? 1.0 : 0.01; // Leaky ReLU
          break;
     case Neuron::Activation::SIGMOID:
          deriv = std::max(0.1, n.output * (1.0 - n.output));
          break;
     default: // LINEAR
          deriv = 1.0;
     }
     return deriv;
}

void Dense::updateWeights(double learningRate)
{
     for (size_t i = 0; i < neurons.size(); ++i)
     {
          for (size_t j = 0; j < neurons[i].getWeights().size(); ++j)
          {
               neurons[i].weights[j] -= learningRate * lastDeltas[i] * lastInput[j];
          }
          neurons[i].bias -= learningRate * lastDeltas[i];
     }
}

void Dense::save(std::ofstream &file) const
{
     if (!file.is_open())
     {
          throw std::runtime_error("Cannot save Dense - file not open");
     }

     file << getType() << " " << neurons.size() << " " << getInputSize() << "\n";
     for (const auto &neuron : neurons)
     {
          for (double weight : neuron.getWeights())
          {
               file << weight << " ";
          }
          file << neuron.bias << " ";

          file << "\n";
     }
}

void Dense::load(std::ifstream &file)
{
     if (!file.is_open())
     {
          throw std::runtime_error("Cannot load Dense - file not open");
     }

     for (size_t i = 0; i < neurons.size(); ++i)
     {
          std::string line;
          if (!std::getline(file, line))
          {
               throw std::runtime_error("Unexpected end of file when reading Dense weights");
          }

          std::istringstream weightStream(line);
          std::vector<double> weights;
          weights.reserve(neurons[i].getWeights().size());

          // Load weights (excluding bias)
          for (size_t j = 0; j < neurons[i].getWeights().size(); ++j)
          {
               double weight;
               if (!(weightStream >> weight))
               {
                    throw std::runtime_error("Failed to parse weight at neuron " + std::to_string(i) + ", weight " + std::to_string(j));
               }
               weights.push_back(weight);
          }

          // Load bias
          double bias;
          if (!(weightStream >> bias))
          {
               throw std::runtime_error("Failed to parse bias at neuron " + std::to_string(i));
          }

          // Apply weights and bias
          neurons[i].setWeights(weights);
          neurons[i].bias = bias;
     }
}
