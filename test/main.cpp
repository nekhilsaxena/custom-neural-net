#include "../src/Model/Model.h"
#include "../src/Model/Components/Neuron/Neuron.h"

#include <iostream>

int main()
{
     try
     {
          Model model;

          // Add layers (dropout only after first layer)
          model.add(10, Neuron::Activation::RELU);      // Input layer
          model.add(10, Neuron::Activation::RELU, 0.2); // Hidden layer with dropout
          model.add(1, Neuron::Activation::SIGMOID);    // Output layer

          // Build the network
          model.build(4);

          // Define sample input and output data
          std::vector<std::vector<double>> inputs = {
              {1.0f, 2.0f, 3.0f, 4.0f},
              {2.0f, 3.0f, 4.0f, 5.0f},
              {3.0f, 4.0f, 5.0f, 6.0f},
              {4.0f, 5.0f, 6.0f, 7.0f}};

          std::vector<double> outputs = {
              10.0f,
              14.0f,
              18.0f,
              22.0f};

          // Train the model
          model.train(inputs, outputs, 2000, 3.6f);

          // Run a prediction
          std::vector<double> testInput = {5.0f, 6.0f, 7.0f, 8.0f};
          std::vector<double> prediction = model.predict(testInput);

          std::vector<double> testInput2 = {0, 0, 0, 0};
          std::vector<double> prediction2 = model.predict(testInput2);

          std::cout << "Prediction: ";
          for (double val : prediction)
               std::cout << val << " ";
          std::cout << "\n";

          std::cout << "Prediction2: ";
          for (double val : prediction2)
               std::cout << val << " ";
          std::cout << "\n";
     }
     catch (const std::exception &e)
     {
          std::cerr << "Error: " << e.what() << "\n";
     }

     return 0;
}
