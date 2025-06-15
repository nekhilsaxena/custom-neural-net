#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <fstream>
#include <numeric>
#include "Model/Operations/Operations.h"

int findFlag(const std::string &flag, int argc, char *argv[])
{
     for (int i = 1; i < argc; ++i)
     {
          if (std::string(argv[i]) == flag)
          {
               return i;
          }
     }
     return -1;
}

int main(int argc, char *argv[])
{
     std::srand(static_cast<unsigned>(std::time(nullptr)));

     int configFlagIndex = findFlag("--config", argc, argv);
     int trainFlagIndex = findFlag("--train", argc, argv);
     int runFlagIndex = findFlag("--run", argc, argv);
     int inputFlagIndex = findFlag("--input", argc, argv);
     int visualizeFlagIndex = findFlag("--visualize", argc, argv);

     if (configFlagIndex != -1 && trainFlagIndex != -1)
     {
          if (runFlagIndex != -1 || inputFlagIndex != -1)
               throw std::invalid_argument("Can't use '--run' or '--input' while training.");

          else
          {
               create(argc, argv, configFlagIndex, trainFlagIndex);
          }
     }
     else if (configFlagIndex != -1 && runFlagIndex != -1)
     {
          if (trainFlagIndex != -1)
               throw std::invalid_argument("Can't use '--train' while running a model.");

          run(argc, argv, configFlagIndex, runFlagIndex, inputFlagIndex);
     }
     else
     {
          throw std::invalid_argument(
              "Usage:\n"
              "  Training: --config <train_config.json> --train <output_dir> [--visualize]\n"
              "  Running:  --config <run_config.json> --run <input_values>\n");
     }

     return 0;
}
