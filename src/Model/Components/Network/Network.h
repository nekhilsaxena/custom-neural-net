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
     std::vector<double> inputMeans;
     std::vector<double> inputStddevs;
     std::vector<std::unique_ptr<Layer>> layers;
     size_t input_dim = 0;

public:
     Network() = default;

     void initialize(int inputSize);

     // Add layer interface
     void add(LayerType type,
              int neuronCount = 0,
              Neuron::Activation::Type activation = Neuron::Activation::RELU,
              double dropoutRate = 0.0);

     // Network operations
     std::vector<double> forward(const std::vector<double> &inputs);
     void backward(const std::vector<double> &expected, double learningRate);
     void train(const std::vector<std::vector<double>> &inputs,
                const std::vector<std::vector<double>> &targets,
                int epochs, double learningRate,
                bool verbose = false,
                std::function<void(float, float)> onEpochEnd = nullptr);

     std::vector<double> predict(std::vector<double> input);

     // Serialization
     void saveWeights(const std::string &filename);
     void loadWeights(const std::string &filename);

     // Mode control
     void setTrainingMode(bool training);

     std::vector<std::vector<double>> normalizeInputs(const std::vector<std::vector<double>> &inputs, std::vector<double> &means, std::vector<double> &stddevs) const;

     std::vector<double> normalizeInput(std::vector<double> inputs);

     // Getters
     size_t getInputSize() const { return input_dim; }
     size_t getOutputSize() const { return layers.empty() ? 0 : layers.back()->getOutputSize(); }
};

#endif