#ifndef MODEL_H
#define MODEL_H

#include "./Components/Network/Network.h"
#include <vector>
#include <memory>
#include <stdexcept>

class Model
{
public:
     Model() = default;

     // Add a dense layer with optional dropout
     void add(int neurons, Neuron::Activation::Type activation = Neuron::Activation::RELU);
     void add(double dropoutRate);

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
          int neurons;
          Neuron::Activation::Type activation;
          double dropoutRate;
     };

     std::unique_ptr<Network> network;
     std::vector<LayerConfig> layers;
     bool built = false;
};

#endif