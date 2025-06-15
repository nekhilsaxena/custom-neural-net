#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include "../Components/Network/Network.h"
#include "../Utils/Utils.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

std::vector<double> convertToInputs(char *argv[], int startIndex, int argc)
{
     std::vector<double> inputs;
     for (int i = startIndex; i < argc; ++i)
     {
          try
          {
               inputs.push_back(std::stod(argv[i]));
          }
          catch (...)
          {
               break;
          }
     }
     return inputs;
}
int run(int argc, char *argv[], int configFlagIndex, int runFlagIndex, int inputFlagIndex)
{
     if (configFlagIndex == -1 || inputFlagIndex == -1)
          throw std::invalid_argument("Missing required flags '--config' and '--input'.");

     // Step 1: Read config file
     std::string configPath = argv[configFlagIndex + 1];
     std::ifstream configFile(configPath);
     if (!configFile)
          throw std::runtime_error("Failed to open config file: " + configPath);

     json config;
     configFile >> config;

     if (!config.contains("type") || config["type"] != "run")
          throw std::invalid_argument("Config file must have \"type\": \"run\".");

     std::cout << "*** Parsing Run Config ***\n";

     // Step 2: Load model weights
     std::string modelPath = config["model"]["path"];
     std::cout << "Model Path: " << modelPath << "\n";

     Network net({});
     net.loadWeights(modelPath);

     // Step 3: Extract input fields and expected input size
     int expectedInputCount = config["inputs"]["count"];
     std::vector<std::string> inputNames = config["inputs"]["names"];

     std::vector<double> input = convertToInputs(argv, inputFlagIndex + 1, argc);
     if (input.size() != static_cast<size_t>(expectedInputCount))
          throw std::runtime_error("Provided input size does not match expected input count in config.");

     std::cout << "*** Input Size is Valid ***\n\n";

     // Step 4: Run forward pass
     auto output = net.forward(input);

     std::cout << "Input: ";
     for (double d : input)
          std::cout << d << " ";
     std::cout << "\nOutput: ";
     for (double o : output)
          std::cout << o << " ";
     std::cout << "\n";

     return 0;
}