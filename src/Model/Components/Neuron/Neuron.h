#pragma once
#include <vector>
#include <cmath>
#include <numeric>
#include <cstdlib>
#include <string>
#include <functional>

class Neuron
{
public:
     struct Activation
     {
          enum Type
          {
               RELU,
               SIGMOID,
               LINEAR
          };
          Type type;
          std::function<double(double)> function;
          std::function<double(double)> derivative;

          Activation(Type t) : type(t)
          {
               switch (t)
               {
               case RELU:
                    function = [this](double x)
                    { return relu(x); };
                    derivative = [this](double x)
                    { return reluDerivative(x); };
                    break;
               case SIGMOID:
                    function = [this](double x)
                    { return sigmoid(x); };
                    derivative = [this](double x)
                    { return sigmoidDerivative(x); };
                    break;
               case LINEAR:
                    function = [](double x)
                    { return x; };
                    derivative = [](double)
                    { return 1.0; };
                    break;
               }
          }
     };
     std::vector<double> weights;
     double bias;
     double z;          // Weighted input (before activation)
     double output;     // After activation
     double lastOutput; // For backprop
     double delta;      // Backprop error
     bool isOutputNeuron;

     Neuron(int numInputs, bool isOutputNeuron, Activation::Type activation = Activation::RELU);

     void setActivation(Activation::Type type);

     Activation::Type activation_;

     // Activation functions
     static double relu(double x);
     static double reluDerivative(double x);
     static double sigmoid(double x);
     static double sigmoidDerivative(double x);

     // Forward pass
     double activate(double x);
     double forward(const std::vector<double> &inputs);

     // Backprop
     void computeOutputDelta(double target);
     void computeHiddenDelta(const std::vector<Neuron> &nextLayer, int index);
     void updateWeights(const std::vector<double> &inputs, double learningRate);

     // Save/load
     void saveWeights(std::ofstream &file, size_t layerIndex, size_t neuronIndex) const;
     void setAll(const std::vector<double> &weightsWithoutBias, double biasVal);

     // Getters
     std::vector<double> getWeights() const;
     double getBias() const;
     void setWeights(std::vector<double> newWeights);
     Activation::Type getActivation() const;
};