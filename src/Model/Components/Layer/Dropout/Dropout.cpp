#include "Dropout.h"
#include <stdexcept>
#include <random>

Dropout::Dropout(double dropoutPercentage)
    : Layer(), // Base constructor
      dropoutRate(1.0 - dropoutPercentage),
      isTraining(true),
      generator(std::random_device{}()),
      distribution(0.0, 1.0)
{
     isTrainable = false; // Dropout layers aren't trainable

     if (dropoutPercentage < 0.0 || dropoutPercentage >= 1.0)
     {
          throw std::invalid_argument("Dropout percentage must be in range [0, 1)");
     }
}

std::vector<double> Dropout::forward(const std::vector<double> &inputs)
{
     lastInput = inputs;
     lastOutput = inputs; // Default to same as input

     if (!isTraining)
     {
          // During inference, scale the inputs by the dropout rate
          for (double &val : lastOutput)
          {
               val *= dropoutRate;
          }
          return lastOutput;
     }

     // During training, apply dropout mask
     mask.resize(inputs.size());
     double scale = 1.0 / dropoutRate;

     for (size_t i = 0; i < inputs.size(); ++i)
     {
          if (distribution(generator) < dropoutRate)
          {
               // Keep this input
               lastOutput[i] = inputs[i] * scale;
               mask[i] = true;
          }
          else
          {
               // Drop this input
               lastOutput[i] = 0.0;
               mask[i] = false;
          }
     }

     return lastOutput;
}

std::vector<double> Dropout::backward(const std::vector<double> &gradients)
{
     if (gradients.size() != lastInput.size())
     {
          throw std::invalid_argument("Gradient size mismatch in Dropout backward pass");
     }

     if (!isTraining)
     {
          // During inference, scale the gradients by dropout rate
          std::vector<double> outputGradients = gradients;
          for (double &grad : outputGradients)
          {
               grad *= dropoutRate;
          }
          return outputGradients;
     }

     // During training, only pass gradients for kept neurons
     std::vector<double> outputGradients(gradients.size(), 0.0);
     double scale = 1.0 / dropoutRate;

     for (size_t i = 0; i < gradients.size(); ++i)
     {
          if (mask[i])
          {
               outputGradients[i] = gradients[i] * scale;
          }
     }

     return outputGradients;
}

void Dropout::save(std::ofstream &file) const
{
     if (!file.is_open())
     {
          throw std::runtime_error("Cannot save Dropout - file not open");
     }
     file << getType() << " " << (1.0 - dropoutRate) << "\n";
}

void Dropout::load(std::ifstream &file)
{
     if (!file.is_open())
     {
          throw std::runtime_error("Cannot load Dropout - file not open");
     }

     std::string type;
     double dropoutPercentage;
     file >> type >> dropoutPercentage;

     if (type != getType())
     {
          throw std::runtime_error("Layer type mismatch in Dropout load");
     }

     dropoutRate = 1.0 - dropoutPercentage;
}