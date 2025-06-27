#include "Dense.h"
#include "../../Neuron/Neuron.h"
#include <stdexcept>
#include <cmath>

Dense::Dense(int numNeurons, int numInputsPerNeuron,
             Neuron::Activation::Type activation, bool isOutputLayer)
    : Layer(), // Base constructor
      lastDeltas()
{
     isTrainable = true; // Set base class protected member

     if (numNeurons <= 0 || numInputsPerNeuron <= 0)
     {
          throw std::invalid_argument("Number of neurons and inputs must be positive");
     }

     neurons.reserve(numNeurons);
     for (int i = 0; i < numNeurons; ++i)
     {
          neurons.emplace_back(numInputsPerNeuron, isOutputLayer, activation);
     }
}

std::vector<double> Dense::forward(const std::vector<double> &inputs)
{
     if (inputs.size() != getInputSize())
     {
          throw std::invalid_argument("Input size mismatch in Dense forward pass");
     }

     lastInput = inputs;
     lastOutput.resize(neurons.size());

     for (size_t i = 0; i < neurons.size(); ++i)
     {
          lastOutput[i] = neurons[i].forward(inputs);
     }

     return lastOutput;
}

void Dense::updateWeights(double learningRate)
{
     for (size_t i = 0; i < neurons.size(); ++i)
     {
          neurons[i].updateWeights(lastInput, learningRate);
     }
}

void Dense::save(std::ofstream &file) const
{
     if (!file.is_open())
     {
          throw std::runtime_error("Cannot save Dense - file not open");
     }

     file << getType() << " " << neurons.size() << " " << getInputSize() << "\n";
     for (const auto &neuron : neurons)
     {
          for (double weight : neuron.getWeights())
          {
               file << weight << " ";
          }
          file << "\n";
     }
}

// void Dense::load(std::ifstream &file)
// {
//      if (!file.is_open())
//      {
//           throw std::runtime_error("Cannot load Dense - file not open");
//      }

//      size_t numNeurons, numInputs;
//      std::string type;
//      file >> type >> numNeurons >> numInputs;

//      if (type != getType())
//      {
//           throw std::runtime_error("Layer type mismatch in Dense load");
//      }

//      neurons.clear();
//      neurons.reserve(numNeurons);

//      for (size_t i = 0; i < numNeurons; ++i)
//      {
//           std::vector<double> weights;
//           weights.reserve(numInputs + 1); // +1 for bias

//           for (size_t j = 0; j < numInputs + 1; ++j)
//           {
//                double weight;
//                file >> weight;
//                weights.push_back(weight);
//           }

//           // Assuming default activation for loaded neurons
//           neurons.emplace_back(weights, false, Neuron::Activation::RELU);
//      }
// }