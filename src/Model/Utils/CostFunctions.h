#ifndef COST_FUNCTIONS_H
#define COST_FUNCTIONS_H

#include <vector>
#include <cmath>
#include <stdexcept>

enum class CostType
{
     MSE,
     MAE,
     BCE,
     CrossEntropy,
     Huber,
     LogCosh
};

class CostFunction
{
public:
     static double compute(CostType type, const std::vector<double> &expected, const std::vector<double> &actual);
     static std::vector<double> derivative(CostType type, const std::vector<double> &expected, const std::vector<double> &actual);

private:
     static double mse(const std::vector<double> &expected, const std::vector<double> &actual);
     static std::vector<double> mseDerivative(const std::vector<double> &expected, const std::vector<double> &actual);

     static double mae(const std::vector<double> &expected, const std::vector<double> &actual);
     static std::vector<double> maeDerivative(const std::vector<double> &expected, const std::vector<double> &actual);

     static double bce(const std::vector<double> &expected, const std::vector<double> &actual);
     static std::vector<double> bceDerivative(const std::vector<double> &expected, const std::vector<double> &actual);

     static double crossEntropy(const std::vector<double> &expected, const std::vector<double> &actual);
     static std::vector<double> crossEntropyDerivative(const std::vector<double> &expected, const std::vector<double> &actual);

     static double huber(const std::vector<double> &expected, const std::vector<double> &actual, double delta = 1.0);
     static std::vector<double> huberDerivative(const std::vector<double> &expected, const std::vector<double> &actual, double delta = 1.0);

     static double logCosh(const std::vector<double> &expected, const std::vector<double> &actual);
     static std::vector<double> logCoshDerivative(const std::vector<double> &expected, const std::vector<double> &actual);
};

#endif // COST_FUNCTIONS_H
