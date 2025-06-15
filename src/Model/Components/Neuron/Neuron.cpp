#include "Neuron.h"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <fstream>
#include <numeric>

// Constructor with Xavier/Glorot initialization
Neuron::Neuron(int numInputs, bool isOutputNeuron, Activation::Type activation)
    : isOutputNeuron(isOutputNeuron), activation_(activation)
{
     double range = sqrt(6.0 / (numInputs + 1)); // Xavier/Glorot for ReLU
     for (int i = 0; i < numInputs; ++i)
     {
          weights.push_back(((double)rand() / RAND_MAX) * 2 * range - range);
     }
     bias = 0.0; // Initialize bias to 0
}

void Neuron::setActivation(Activation::Type type)
{
     activation_ = Activation(type);
}

Neuron::Activation Neuron::getActivation() const
{
     return activation_;
}

// ReLU activation
double Neuron::relu(double x)
{
     return x > 0 ? x : 0;
}

// ReLU derivative
double Neuron::reluDerivative(double x)
{
     return x > 0 ? 1 : 0;
}

// Sigmoid activation
double Neuron::sigmoid(double x)
{
     return 1.0 / (1.0 + std::exp(-x));
}

// Derivative of sigmoid
double Neuron::sigmoidDerivative(double x)
{
     double s = sigmoid(x);
     return s * (1 - s);
}

// Activation logic
double Neuron::activate(double x)
{
     if (isOutputNeuron)
     {
          return x; // Linear for regression
     }
     else
     {
          return relu(x); // ReLU for hidden layers
     }
}

// Forward pass
double Neuron::forward(const std::vector<double> &inputs)
{
     if (inputs.size() != weights.size())
     {
          throw std::invalid_argument("Input size does not match number of weights.");
     }

     z = std::inner_product(weights.begin(), weights.end(), inputs.begin(), bias);
     lastOutput = activate(z);
     output = lastOutput;
     return output;
}

// Compute delta for output neuron
void Neuron::computeOutputDelta(double target)
{
     if (isOutputNeuron)
     {
          // For regression (linear output)
          delta = output - target;

          // For classification (uncomment if using sigmoid output):
          // delta = (output - target) * sigmoidDerivative(output);
     }
}

// Compute delta for hidden neuron
void Neuron::computeHiddenDelta(const std::vector<Neuron> &nextLayer, int index)
{
     double sum = 0.0;
     for (const Neuron &next : nextLayer)
     {
          sum += next.weights[index] * next.delta;
     }
     delta = sum * reluDerivative(output); // Use ReLU derivative for hidden layers
}

// Update weights and bias
void Neuron::updateWeights(const std::vector<double> &inputs, double learningRate)
{
     for (size_t i = 0; i < weights.size(); ++i)
     {
          weights[i] -= learningRate * delta * inputs[i];
     }
     bias -= learningRate * delta;
}

// Rest of the functions remain the same as in your original code
void Neuron::saveWeights(std::ofstream &file, size_t layerIndex, size_t neuronIndex) const
{
     file << layerIndex << "," << neuronIndex;
     for (double w : weights)
     {
          file << "," << w;
     }
     file << "," << bias << "\n";
}

std::vector<double> Neuron::getWeights() const { return weights; }
double Neuron::getBias() const { return bias; }

void Neuron::setWeights(std::vector<double> newWeights) { weights = newWeights; }
void Neuron::setBias(double newBias) { bias = newBias; }

void Neuron::setAll(const std::vector<double> &weightsWithoutBias, double biasVal)
{
     weights = weightsWithoutBias;
     bias = biasVal;
}