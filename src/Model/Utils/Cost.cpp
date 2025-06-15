#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <cmath>

double computeMSE(const std::vector<double> &expected, const std::vector<double> &actual)
{
     if (expected.size() != actual.size())
     {
          throw std::invalid_argument("Expected and actual output size mismatch.");
     }

     double sum = 0.0;
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double diff = expected[i] - actual[i];
          sum += diff * diff;
     }

     return sum / expected.size();
}
