#include "../src/Model/Model.h"
#include "../src/Model/Components/Neuron/Neuron.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

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

// Function to read CSV file and return input and output data
std::pair<std::vector<std::vector<double>>, std::vector<double>> readCSV(const std::string &filename,
                                                                         int numInputs,
                                                                         int outputCol = -1)
{
     std::ifstream file(filename);
     if (!file.is_open())
     {
          throw std::runtime_error("Could not open file: " + filename);
     }

     std::vector<std::vector<double>> inputs;
     std::vector<double> outputs;
     std::string line, value;

     // Read header if exists (and skip it)
     std::getline(file, line);

     while (std::getline(file, line))
     {
          std::stringstream ss(line);
          std::vector<double> row;
          int col = 0;
          double outputVal = 0.0;

          while (std::getline(ss, value, ','))
          {
               try
               {
                    double num = std::stod(value);
                    if (outputCol == -1)
                    {
                         // If output column not specified, last column is output
                         if (col < numInputs)
                         {
                              row.push_back(num);
                         }
                         else if (col == numInputs)
                         {
                              outputVal = num;
                         }
                    }
                    else
                    {
                         // Specific output column case
                         if (col == outputCol)
                         {
                              outputVal = num;
                         }
                         else if (row.size() < numInputs)
                         {
                              row.push_back(num);
                         }
                    }
                    col++;
               }
               catch (const std::exception &e)
               {
                    throw std::runtime_error("Error parsing value in CSV: " + value);
               }
          }

          if (row.size() != numInputs)
          {
               throw std::runtime_error("Incorrect number of input features in row");
          }

          inputs.push_back(row);
          outputs.push_back(outputVal);
     }

     return {inputs, outputs};
}

int train()
{
     try
     {
          Model model;

          const std::string csvFilename = "../../Training/data.csv"; // Change to your CSV file path
          const int numInputFeatures = 4;                            // Number of input features
          const int outputColumn = 0;                                // -1 for last column, or specify column index
          const int epochs = 2000;                                   // Training epochs
          const double learningRate = 0.001;                         // Learning rate

          // Best architecture for your financial data
          model.add(32, Neuron::Activation::RELU);  // Dense layer
          model.add(0.2);                           // Dropout layer
          model.add(16, Neuron::Activation::RELU);  // Dense layer
          model.add(0.3);                           // Dropout layer
          model.add(8, Neuron::Activation::RELU);   // Third hidden layer
          model.add(1, Neuron::Activation::LINEAR); // Output layer (predicting Close price)

          // Build the network
          model.build(numInputFeatures);

          // Load data from CSV
          auto [inputs, outputs] = readCSV(csvFilename, numInputFeatures, outputColumn);

          if (inputs.empty() || outputs.empty())
          {
               throw std::runtime_error("No data loaded from CSV file");
          }

          std::cout << "Loaded " << inputs.size() << " samples from CSV file\n";

          // Train the model
          model.train(inputs, outputs, epochs, learningRate);

          // Save the model
          model.save("dropout_model.csv");

          // Test the model with some samples from the training data
          std::cout << "\nTesting with some samples:\n";
          for (size_t i = 0; i < std::min(inputs.size(), static_cast<size_t>(5)); i++)
          {
               auto prediction = model.predict(inputs[i]);
               std::cout << "Input: ";
               for (double val : inputs[i])
               {
                    std::cout << val << " ";
               }
               std::cout << "| True: " << outputs[i] << " | Predicted: " << prediction[0] << "\n";
          }
     }
     catch (const std::exception &e)
     {
          std::cerr << "Error: " << e.what() << "\n";
          return 1;
     }

     return 0;
}

int train2()
{
     try
     {
          Model model;

          const std::string csvFilename = "../../Training/data.csv"; // Change to your CSV file path
          const int numInputFeatures = 4;                            // Number of input features
          const int outputColumn = 0;                                // -1 for last column, or specify column index
          const int epochs = 2000;                                   // Training epochs
          const double learningRate = 0.001;                         // Learning rate

          // Best architecture for your financial data
          model.add(32, Neuron::Activation::RELU); // Dense layer
          // model.add(0.2);                           // Dropout layer
          model.add(16, Neuron::Activation::RELU); // Dense layer
          // model.add(0.3);                           // Dropout layer
          model.add(8, Neuron::Activation::RELU);   // Third hidden layer
          model.add(1, Neuron::Activation::LINEAR); // Output layer (predicting Close price)

          // Build the network
          model.build(numInputFeatures);

          // Load data from CSV
          auto [inputs, outputs] = readCSV(csvFilename, numInputFeatures, outputColumn);

          if (inputs.empty() || outputs.empty())
          {
               throw std::runtime_error("No data loaded from CSV file");
          }

          std::cout << "Loaded " << inputs.size() << " samples from CSV file\n";

          // Train the model
          model.train(inputs, outputs, epochs, learningRate);

          // Save the model
          model.save("no_dropout_model.csv");

          // Test the model with some samples from the training data
          std::cout << "\nTesting with some samples:\n";
          for (size_t i = 0; i < std::min(inputs.size(), static_cast<size_t>(5)); i++)
          {
               auto prediction = model.predict(inputs[i]);
               std::cout << "Input: ";
               for (double val : inputs[i])
               {
                    std::cout << val << " ";
               }
               std::cout << "| True: " << outputs[i] << " | Predicted: " << prediction[0] << "\n";
          }
     }
     catch (const std::exception &e)
     {
          std::cerr << "Error: " << e.what() << "\n";
          return 1;
     }

     return 0;
}

int test(std::string modelPath)
{
     try
     {
          const std::string csvFilename = "../../Training/data.csv"; // Change to your CSV file path
          const int numInputFeatures = 4;                            // Number of input features
          const int outputColumn = 0;                                // -1 for last column, or specify column index

          // Load data from CSV
          auto [inputs, outputs] = readCSV(csvFilename, numInputFeatures, outputColumn);

          Model model;
          std::cout << "model loading" << "\n";
          model.load("../../model.csv");

          // Test the model with some samples from the training data
          std::cout << "\nTesting with some samples:\n";
          for (size_t i = 0; i < std::min(inputs.size(), static_cast<size_t>(5)); i++)
          {
               auto prediction = model.predict(inputs[i]);
               std::cout << "Input: ";
               for (double val : inputs[i])
               {
                    std::cout << val << " ";
               }
               std::cout << "| True: " << outputs[i] << " | Predicted: " << prediction[0] << "\n";
          }
     }
     catch (const std::exception &e)
     {
          std::cerr << "Error: " << e.what() << "\n";
          return 1;
     }

     return 0;
}

int main(int argc, char *argv[])
{
     int train1Flag = findFlag("--train1", argc, argv);
     int train2Flag = findFlag("--train2", argc, argv);
     int test1Flag = findFlag("--test1", argc, argv);
     int test2Flag = findFlag("--test2", argc, argv);
     if (train1Flag != -1)
          train();
     else if (train2Flag != -1)
          train2();
     else if (test1Flag != -1)
          test("../../dropout_model");
     else if (test2Flag != -1)
          test("../../no_dropout_model");
     return 0;
}
