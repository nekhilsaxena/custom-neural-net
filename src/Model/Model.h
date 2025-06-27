#ifndef MODEL_H
#define MODEL_H

#include "Network.h"
#include <vector>
#include <memory>
#include <stdexcept>

class Model
{
public:
     Model() = default;

     // Add a dense layer with optional dropout
     void add(int units,
              Neuron::Activation::Type activation = Neuron::Activation::RELU,
              double dropoutRate = 0.0);

     // Build the network with input shape
     void build(int inputSize);

     // Training and prediction
     void train(const std::vector<std::vector<double>> &inputs,
                const std::vector<double> &outputs,
                int epochs,
                double learningRate);

     std::vector<double> predict(const std::vector<double> &input);
     void save(const std::string &filename);
     void load(const std::string &filename);

private:
     struct LayerConfig
     {
          Network::LayerType type;
          int units;
          Neuron::Activation::Type activation;
          double dropoutRate;
          bool isOutput;
     };

     std::unique_ptr<Network> network;
     std::vector<LayerConfig> layers;
     bool built = false;
};

#endif