#ifndef NETWORK_H
#define NETWORK_H

#include "../Layer/Layer.h"
#include "../Layer/Dense/Dense.h"
#include "../Layer/Dropout/Dropout.h"
#include <vector>
#include <string>
#include <memory>
#include <functional>

class Network
{
public:
     enum class LayerType
     {
          DENSE,
          DROPOUT
     };

private:
     std::vector<std::unique_ptr<Layer>> layers;
     size_t nextLayerInputSize = 0; // Tracks input size for next layer

public:
     Network() = default;

     // Add layer interface
     void add(LayerType type,
              int neuronCount = 0,
              Neuron::Activation::Type activation = Neuron::Activation::RELU,
              double dropoutRate = 0.0,
              bool isOutputLayer = false);

     // Network operations
     std::vector<double> forward(const std::vector<double> &inputs);
     void backward(const std::vector<double> &expected, double learningRate);
     void train(const std::vector<std::vector<double>> &inputs,
                const std::vector<std::vector<double>> &targets,
                int epochs, double learningRate,
                bool verbose = false,
                std::function<void(float, float)> onEpochEnd = nullptr);

     // Serialization
     void saveWeights(const std::string &filename);
     void loadWeights(const std::string &filename);

     // Mode control
     void setTrainingMode(bool training);

     // Getters
     size_t getInputSize() const { return layers.empty() ? 0 : layers.front()->getInputSize(); }
     size_t getOutputSize() const { return layers.empty() ? 0 : layers.back()->getOutputSize(); }
};

#endif