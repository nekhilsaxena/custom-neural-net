#ifndef MODEL_H
#define MODEL_H

#include "Components/Network/Network.h"
#include "Components/Neuron/Neuron.h"

#include <vector>
#include <string>
#include <fstream>
#include <functional>
#include <memory>

class Model
{
private:
     std::vector<int> layerSizes;
     std::vector<Neuron::Activation::Type> activations;
     std::unique_ptr<Network> network;

public:
     Model();

     void add(int neurons, Neuron::Activation::Type activation);
     void build();
     void train(const std::vector<std::vector<double>> &inputs, const std::vector<double> &outputs, int epochs = 100, double learningRate = 0.01);
     std::vector<double> predict(const std::vector<double> &input);
};

#endif
