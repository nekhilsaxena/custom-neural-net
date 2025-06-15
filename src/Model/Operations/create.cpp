#include <nlohmann/json.hpp>
#include <iostream>
#include <filesystem>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <string>
#include <stdexcept>
#include "../Components/Network/Network.h"
#include "../Components/Neuron/Neuron.h"
#include "../Utils/Utils.h"
#include "../Visualization/Visualization.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

std::vector<int> convertToLayers(char *items[], int start, int end)
{
     std::vector<int> result;
     for (int i = start; i < end; ++i)
     {
          result.push_back(std::stoi(items[i]));
     }
     return result;
}

std::string convertToSavePath(char *items[], int index)
{
     return items[index] ? items[index] : "";
}

int create(int argc, char *argv[], int configFlagIndex, int trainFlagIndex)
{
     if (configFlagIndex == -1 || trainFlagIndex == -1)
          throw std::invalid_argument("Missing required flags '--config' and '--train'.");

     // Load JSON config
     std::string configPath = argv[configFlagIndex + 1];
     std::ifstream configFile(configPath);
     if (!configFile)
          throw std::runtime_error("Failed to open config file: " + configPath);

     json config;
     configFile >> config;

     if (!config.contains("type") || config["type"] != "training")
          throw std::invalid_argument("The config file must have \"type\": \"training\".");

     std::cout << "*** Parsing Config ***\n";
     std::string csvPath = config["data"]["path"];

     std::vector<std::string> inputNames = config["inputs"]["names"].get<std::vector<std::string>>();
     std::vector<std::string> outputNames = config["outputs"]["names"].get<std::vector<std::string>>();

     std::vector<int> layers = config["network"]["layers"].get<std::vector<int>>();
     Neuron::Activation::Type hiddenActivation = config["network"]["activation"]["hidden"].get<std::string>() == "RELU" ? Neuron::Activation::RELU : config["network"]["activation"]["hidden"].get<std::string>() == "SIGMOID" ? Neuron::Activation::SIGMOID
                                                                                                                                                                                                                               : Neuron::Activation::LINEAR;
     Neuron::Activation::Type outputActivation = config["network"]["activation"]["output"].get<std::string>() == "RELU" ? Neuron::Activation::RELU : config["network"]["activation"]["output"].get<std::string>() == "SIGMOID" ? Neuron::Activation::SIGMOID
                                                                                                                                                                                                                               : Neuron::Activation::LINEAR;

     int epochs = config["network"]["training"]["epochs"].get<int>();
     double learningRate = config["network"]["training"]["learning_rate"].get<double>();

     std::string trainingSavePath = config["build"]["network"]["path"];
     std::string runConfigSavePath = config["build"]["runConfig"]["path"];

     json inputSection = config["inputs"];
     json outputSection = config["outputs"];

     std::cout << "  Data Path: " << csvPath << "\n";
     std::cout << "  Inputs: ";
     for (const auto &n : inputNames)
          std::cout << n << " ";
     std::cout << "\n  Outputs: ";
     for (const auto &n : outputNames)
          std::cout << n << " ";
     std::cout << "\n  Layers: ";
     for (int l : layers)
          std::cout << l << " ";
     std::cout << "\n";

     std::string absolutePath = std::filesystem::absolute(csvPath).string();

     // Read CSV (with headers)
     std::vector<std::vector<std::string>> rawCSV = readRawCSV(absolutePath);
     if (rawCSV.empty())
          throw std::runtime_error("CSV file is empty.");

     std::vector<std::string> headers = rawCSV[0];
     std::map<std::string, size_t> headerIndex;
     for (size_t i = 0; i < headers.size(); ++i)
          headerIndex[headers[i]] = i;

     // Extract inputs and expected outputs
     std::vector<std::vector<double>> inputs;
     std::vector<std::vector<double>> targets;

     for (size_t i = 1; i < rawCSV.size(); ++i)
     {
          const auto &row = rawCSV[i];
          std::vector<double> inputRow;
          std::vector<double> outputRow;

          for (const std::string &name : inputNames)
          {
               if (headerIndex.find(name) == headerIndex.end())
                    throw std::runtime_error("Missing input column in CSV: " + name);
               inputRow.push_back(std::stod(row[headerIndex[name]]));
          }

          for (const std::string &name : outputNames)
          {
               if (headerIndex.find(name) == headerIndex.end())
                    throw std::runtime_error("Missing output column in CSV: " + name);
               outputRow.push_back(std::stod(row[headerIndex[name]]));
          }

          inputs.push_back(inputRow);
          targets.push_back(outputRow);
     }

     if (!fs::exists(trainingSavePath))
     {
          std::cout << "Creating save directory...\n";
          fs::create_directories(trainingSavePath);
     }

     // --- 1. Create the network
     Network net(layers, Neuron::Activation::RELU, Neuron::Activation::SIGMOID);

     // --- 2. Train it
     net.train(inputs, targets, epochs, learningRate, true);

     // --- 3. Evaluate it after training
     double totalCost = 0.0;

     for (size_t i = 0; i < inputs.size(); ++i)
     {
          const std::vector<double> &input = inputs[i];
          const std::vector<double> &expected = targets[i];

          std::vector<double> output = net.forward(input);
          double cost = computeMSE(expected, output);
          totalCost += cost;

          std::cout << "Input: ";
          for (double d : input)
               std::cout << d << " ";
          std::cout << "\nExpected: ";
          for (double d : expected)
               std::cout << d << " ";
          std::cout << "\nActual: ";
          for (double o : output)
               std::cout << o << " ";
          std::cout << "\nCost: " << cost << "\n---\n";
     }

     double averageCost = totalCost / inputs.size();
     std::cout << "\nAverage Cost over Dataset: " << averageCost << "\n";

     // Save weights
     std::string weightsPath = trainingSavePath + "/weights.csv";
     std::cout << "Saving network weights...\n";
     net.saveWeights(weightsPath);
     std::cout << "Training complete.\n";

     // Create run_config
     json runConfig;
     runConfig["type"] = "run";
     runConfig["model"] = {
         {"type", "CSV"},
         {"path", weightsPath}};
     runConfig["inputs"] = inputSection;
     runConfig["outputs"] = outputSection;
     runConfig["network"] = config["network"];

     std::ofstream out(runConfigSavePath + "/run_config.json");
     out << runConfig.dump(4);
     std::cout << "Generated run_config.json at: " << runConfigSavePath + "/run_config.json" << "\n";

     return 0;
}