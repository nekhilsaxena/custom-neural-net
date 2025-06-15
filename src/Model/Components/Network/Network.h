#ifndef NETWORK_H
#define NETWORK_H

#include "../Layer/Layer.h"
#include "../Neuron/Neuron.h"
#include <vector>
#include <string>
#include <fstream>
#include <functional>

class Network
{
private:
     std::vector<Layer> layers;
     std::vector<int> layerSizes;

public:
     Network(const std::vector<int> &layerSizes,
             Neuron::Activation::Type hiddenActivation = Neuron::Activation::RELU,
             Neuron::Activation::Type outputActivation = Neuron::Activation::LINEAR);

     std::vector<double> forward(const std::vector<double> &inputs);
     void saveWeights(const std::string &filename) const;
     void loadWeights(const std::string &filename);

     std::vector<int> getLayerSizes();

     // Backpropagation
     void backward(const std::vector<double> &expected, double learningRate);

     void train(const std::vector<std::vector<double>> &inputs, const std::vector<std::vector<double>> &targets, int epochs, double learningRate, bool verbose, std::function<void(float, float)> onEpochEnd = nullptr);
};

#endif
