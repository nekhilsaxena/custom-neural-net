#include "Network.h"
#include <iostream>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <functional>
#include <memory>

void Network::saveWeights(const std::string &filename)
{
     std::ofstream file(filename);
     if (!file.is_open())
     {
          throw std::runtime_error("Unable to open file to save weights.");
     }

     // Save normalization stats
     file << "Normalization:\n";

     file << "Means ";
     for (const auto &mean : inputMeans)
     {
          file << mean << " ";
     }
     file << "\n";

     file << "Stddevs ";
     for (const auto &stddev : inputStddevs)
     {
          file << stddev << " ";
     }
     file << "\n";

     // Save layer weights and configurations
     file << "Weights:\n";
     for (const auto &layer : layers)
     {
          if (layer->getType() == "Dense" || layer->getType() == "Dropout")
          {
               layer->save(file);
          }
          else
          {
               throw std::runtime_error("Unknown layer type during save: " + layer->getType());
          }
     }
}

void Network::loadWeights(const std::string &filename)
{
     std::ifstream file(filename);
     if (!file)
     {
          throw std::runtime_error("Failed to open weights file: " + filename);
     }

     std::string line;
     bool readingWeights = false;
     layers.clear();

     while (std::getline(file, line))
     {
          if (line.empty())
               continue;

          if (line == "Normalization:")
          {
               // Read Means
               std::getline(file, line);
               std::istringstream meanStream(line);
               std::string meansLabel;
               meanStream >> meansLabel;
               if (meansLabel != "Means")
                    throw std::runtime_error("Expected 'Means' label");
               inputMeans.clear();
               double value;
               while (meanStream >> value)
                    inputMeans.push_back(value);

               // Read Stddevs
               std::getline(file, line);
               std::istringstream stddevStream(line);
               std::string stddevsLabel;
               stddevStream >> stddevsLabel;
               if (stddevsLabel != "Stddevs")
                    throw std::runtime_error("Expected 'Stddevs' label");
               inputStddevs.clear();
               while (stddevStream >> value)
                    inputStddevs.push_back(value);

               continue;
          }

          if (line == "Weights:")
          {
               readingWeights = true;
               continue;
          }

          if (!readingWeights)
               continue;

          // Detect layer type
          std::istringstream iss(line);
          std::string layerType;
          iss >> layerType;

          if (layerType == "Dense")
          {
               size_t numNeurons, numInputs;
               iss >> numNeurons >> numInputs;

               if (layers.empty())
               {
                    this->initialize(numInputs);
               }

               auto dense = std::make_unique<Dense>(numNeurons, numInputs, Neuron::Activation::RELU);
               dense->load(file);
               layers.push_back(std::move(dense));
          }
          else if (layerType == "Dropout")
          {
               float dropoutRate;
               iss >> dropoutRate;

               int prevOutputSize = layers.empty() ? 0 : layers.back()->getOutputSize();
               auto dropout = std::make_unique<Dropout>(dropoutRate, prevOutputSize);

               layers.push_back(std::move(dropout));
               std::cout << "dropout layer added \n";
          }
          else
          {
               throw std::runtime_error("Unknown layer type: " + layerType);
          }
     }
}

std::vector<std::vector<double>> Network::normalizeInputs(const std::vector<std::vector<double>> &inputs,
                                                          std::vector<double> &means,
                                                          std::vector<double> &stddevs) const
{
     if (inputs.empty())
     {
          throw std::invalid_argument("Inputs cannot be empty");
     }

     size_t numFeatures = inputs[0].size();
     means.assign(numFeatures, 0.0);
     stddevs.assign(numFeatures, 0.0);

     // Compute means
     for (size_t j = 0; j < numFeatures; ++j)
     {
          double sum = 0.0;
          for (const auto &sample : inputs)
          {
               sum += sample[j];
          }
          means[j] = sum / inputs.size();
     }

     // Compute standard deviations
     for (size_t j = 0; j < numFeatures; ++j)
     {
          double sqSum = 0.0;
          for (const auto &sample : inputs)
          {
               sqSum += (sample[j] - means[j]) * (sample[j] - means[j]);
          }
          stddevs[j] = std::sqrt(sqSum / inputs.size()) + 1e-8; // Add epsilon to avoid division by zero
     }

     // Normalize inputs
     std::vector<std::vector<double>> normalized = inputs;
     for (auto &sample : normalized)
     {
          for (size_t j = 0; j < numFeatures; ++j)
          {
               sample[j] = (sample[j] - means[j]) / stddevs[j];
          }
     }

     return normalized;
}

std::vector<double> Network::normalizeInput(std::vector<double> input)
{
     if (input.size() != inputMeans.size())
     {
          throw std::invalid_argument("Input size does not match training input size for normalization");
     }

     std::vector<double> normalized = input;
     for (size_t j = 0; j < normalized.size(); ++j)
     {
          normalized[j] = (normalized[j] - inputMeans[j]) / inputStddevs[j];
     }
     return normalized;
}

void Network::setTrainingMode(bool training)
{
     for (auto &layer : layers)
     {
          if (auto dropout = dynamic_cast<Dropout *>(layer.get()))
          {
               dropout->setTraining(training);
          }
     }
}