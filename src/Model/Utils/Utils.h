#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include <string>
#include <functional>

struct StatsSummary
{
     float mean;
     float median;
     float mode;
     float q1;
     float q3;
     float min; // excluding outliers
     float max; // excluding outliers
};

struct Cost
{
     enum Type
     {
          MSE,
          BCE
          // Add more types here later
     };

     Type type;
     std::function<double(const std::vector<double> &, const std::vector<double> &)> function;
     std::function<std::vector<double>(const std::vector<double> &, const std::vector<double> &)> derivative;

     Cost(Type t) : type(t)
     {
          switch (t)
          {
          case MSE:
               function = [](const std::vector<double> &expected, const std::vector<double> &actual)
               {
                    double sum = 0.0;
                    for (size_t i = 0; i < expected.size(); ++i)
                         sum += std::pow(expected[i] - actual[i], 2);
                    return sum / expected.size();
               };
               derivative = [](const std::vector<double> &expected, const std::vector<double> &actual)
               {
                    std::vector<double> grad(expected.size());
                    for (size_t i = 0; i < expected.size(); ++i)
                         grad[i] = 2.0 * (actual[i] - expected[i]) / expected.size();
                    return grad;
               };
               break;

          case BCE:
               function = [](const std::vector<double> &expected, const std::vector<double> &actual)
               {
                    double sum = 0.0;
                    for (size_t i = 0; i < expected.size(); ++i)
                    {
                         double a = std::min(std::max(actual[i], 1e-12), 1.0 - 1e-12); // Clamp to avoid log(0)
                         sum += -(expected[i] * std::log(a) + (1.0 - expected[i]) * std::log(1.0 - a));
                    }
                    return sum / expected.size();
               };
               derivative = [](const std::vector<double> &expected, const std::vector<double> &actual)
               {
                    std::vector<double> grad(expected.size());
                    for (size_t i = 0; i < expected.size(); ++i)
                    {
                         double a = std::min(std::max(actual[i], 1e-12), 1.0 - 1e-12);
                         grad[i] = (a - expected[i]) / (a * (1 - a) * expected.size());
                    }
                    return grad;
               };
               break;
          }
     }
};

StatsSummary computeStatistics(const std::vector<float> &data);

#endif
