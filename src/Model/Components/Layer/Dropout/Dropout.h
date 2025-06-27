#ifndef DROPOUT_H
#define DROPOUT_H

#include "../Layer.h"
#include <random>
#include <vector>

class Dropout : public Layer
{
private:
     double dropoutRate; // Probability of keeping a neuron
     bool isTraining;
     std::vector<bool> mask;
     std::mt19937 generator;
     std::uniform_real_distribution<double> distribution;

public:
     // Constructor
     Dropout(double dropoutPercentage = 0.5);

     // Training mode control
     void setTraining(bool training) { isTraining = training; }

     // Layer interface implementation
     std::vector<double> forward(const std::vector<double> &inputs) override;
     std::vector<double> backward(const std::vector<double> &gradients) override;
     std::string getType() const override { return "Dropout"; }
     size_t getOutputSize() const override { return getInputSize(); }
     size_t getInputSize() const override
     {
          return lastInput.empty() ? 0 : lastInput.size();
     }
     void save(std::ofstream &file) const override;
     void load(std::ifstream &file) override;

     // Debug access
     const std::vector<bool> &getLastMask() const { return mask; }
};

#endif