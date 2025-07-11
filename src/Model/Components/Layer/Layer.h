#ifndef LAYER_H
#define LAYER_H

#include <vector>
#include <string>
#include <fstream>

class Layer
{
protected:
     std::vector<double> lastInput;
     std::vector<double> lastOutput;
     bool isTrainable;

public:
     virtual ~Layer() = default;

     // Forward pass
     virtual std::vector<double> forward(const std::vector<double> &inputs) = 0;

     // Backward pass
     virtual std::vector<double> backward(const std::vector<double> &gradients, const std::vector<std::vector<double>> &nextLayerWeights = {})
     {
          return gradients;
     }

     // Update weights
     virtual void updateWeights(double learningRate) {}

     // Serialization
     virtual void save(std::ofstream &file) const = 0;
     virtual void load(std::ifstream &file) = 0;

     // Layer info
     virtual std::string getType() const = 0;
     virtual size_t getOutputSize() const = 0;
     virtual size_t getInputSize() const = 0;

     // Training control
     bool isTrainableLayer() const { return isTrainable; }
     void setIsTrainableLayer(bool trainable) { isTrainable = trainable; }
     const std::vector<double> &getLastOutput() const { return lastOutput; }
};

#endif