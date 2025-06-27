#ifndef DENSE_H
#define DENSE_H

#include "../Layer.h"
#include "../../Neuron/Neuron.h"
#include <vector>

class Dense : public Layer
{
private:
     std::vector<Neuron> neurons;
     std::vector<double> lastDeltas;

public:
     // Constructor
     Dense(int numNeurons, int numInputsPerNeuron,
           Neuron::Activation::Type activation,
           bool isOutputLayer = false);

     // Layer interface implementation
     std::vector<double> forward(const std::vector<double> &inputs) override;
     // std::vector<double> backward(const std::vector<double> &gradients) override;
     void updateWeights(double learningRate) override;
     void save(std::ofstream &file) const override;
     void load(std::ifstream &file) override { file; };
     std::string getType() const override { return "Dense"; }
     size_t getOutputSize() const override { return neurons.size(); }
     size_t getInputSize() const override
     {
          return neurons.empty() ? 0 : neurons[0].getWeights().size() - 1;
     }

     // Neuron access
     std::vector<Neuron> &getNeurons() { return neurons; }
     const std::vector<Neuron> &getNeurons() const { return neurons; }
     Neuron &getNeuron(int index) { return neurons[index]; }
     const Neuron &getNeuron(int index) const { return neurons[index]; }

     // Layer modification
     void addNeuron(const Neuron &neuron) { neurons.push_back(neuron); }
     size_t size() const { return neurons.size(); }
};

#endif