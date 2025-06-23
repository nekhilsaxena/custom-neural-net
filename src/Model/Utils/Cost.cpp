#include "CostFunctions.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <numeric>

double CostFunction::compute(CostType type, const std::vector<double> &expected, const std::vector<double> &actual)
{
     switch (type)
     {
     case CostType::MSE:
          return mse(expected, actual);
     case CostType::MAE:
          return mae(expected, actual);
     case CostType::BCE:
          return bce(expected, actual);
     case CostType::CrossEntropy:
          return crossEntropy(expected, actual);
     case CostType::Huber:
          return huber(expected, actual);
     case CostType::LogCosh:
          return logCosh(expected, actual);
     default:
          throw std::invalid_argument("Unknown cost type");
     }
}

std::vector<double> CostFunction::derivative(CostType type, const std::vector<double> &expected, const std::vector<double> &actual)
{
     switch (type)
     {
     case CostType::MSE:
          return mseDerivative(expected, actual);
     case CostType::MAE:
          return maeDerivative(expected, actual);
     case CostType::BCE:
          return bceDerivative(expected, actual);
     case CostType::CrossEntropy:
          return crossEntropyDerivative(expected, actual);
     case CostType::Huber:
          return huberDerivative(expected, actual);
     case CostType::LogCosh:
          return logCoshDerivative(expected, actual);
     default:
          throw std::invalid_argument("Unknown cost type");
     }
}

// --- MSE ---
double CostFunction::mse(const std::vector<double> &expected, const std::vector<double> &actual)
{
     double sum = 0.0;
     for (size_t i = 0; i < expected.size(); ++i)
          sum += std::pow(expected[i] - actual[i], 2);
     return sum / expected.size();
}

std::vector<double> CostFunction::mseDerivative(const std::vector<double> &expected, const std::vector<double> &actual)
{
     std::vector<double> grad(expected.size());
     for (size_t i = 0; i < expected.size(); ++i)
          grad[i] = 2.0 * (actual[i] - expected[i]) / expected.size();
     return grad;
}

// --- MAE ---
double CostFunction::mae(const std::vector<double> &expected, const std::vector<double> &actual)
{
     double sum = 0.0;
     for (size_t i = 0; i < expected.size(); ++i)
          sum += std::abs(expected[i] - actual[i]);
     return sum / expected.size();
}

std::vector<double> CostFunction::maeDerivative(const std::vector<double> &expected, const std::vector<double> &actual)
{
     std::vector<double> grad(expected.size());
     for (size_t i = 0; i < expected.size(); ++i)
          grad[i] = (actual[i] > expected[i]) ? 1.0 : (actual[i] < expected[i] ? -1.0 : 0.0);
     return grad;
}

// --- Binary Cross Entropy ---
double CostFunction::bce(const std::vector<double> &expected, const std::vector<double> &actual)
{
     double sum = 0.0;
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double p = std::clamp(actual[i], 1e-15, 1.0 - 1e-15);
          sum += -expected[i] * std::log(p) - (1.0 - expected[i]) * std::log(1.0 - p);
     }
     return sum / expected.size();
}

std::vector<double> CostFunction::bceDerivative(const std::vector<double> &expected, const std::vector<double> &actual)
{
     std::vector<double> grad(expected.size());
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double p = std::clamp(actual[i], 1e-15, 1.0 - 1e-15);
          grad[i] = (p - expected[i]) / (p * (1.0 - p) * expected.size());
     }
     return grad;
}

// --- Cross Entropy (for multi-class) ---
double CostFunction::crossEntropy(const std::vector<double> &expected, const std::vector<double> &actual)
{
     double sum = 0.0;
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double p = std::clamp(actual[i], 1e-15, 1.0);
          sum += -expected[i] * std::log(p);
     }
     return sum;
}

std::vector<double> CostFunction::crossEntropyDerivative(const std::vector<double> &expected, const std::vector<double> &actual)
{
     std::vector<double> grad(expected.size());
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double p = std::clamp(actual[i], 1e-15, 1.0);
          grad[i] = -expected[i] / p;
     }
     return grad;
}

// --- Huber Loss ---
double CostFunction::huber(const std::vector<double> &expected, const std::vector<double> &actual, double delta)
{
     double sum = 0.0;
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double diff = expected[i] - actual[i];
          if (std::abs(diff) <= delta)
               sum += 0.5 * diff * diff;
          else
               sum += delta * (std::abs(diff) - 0.5 * delta);
     }
     return sum / expected.size();
}

std::vector<double> CostFunction::huberDerivative(const std::vector<double> &expected, const std::vector<double> &actual, double delta)
{
     std::vector<double> grad(expected.size());
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double diff = actual[i] - expected[i];
          if (std::abs(diff) <= delta)
               grad[i] = diff;
          else
               grad[i] = delta * ((diff > 0) ? 1.0 : -1.0);
     }
     return grad;
}

// --- Log-Cosh Loss ---
double CostFunction::logCosh(const std::vector<double> &expected, const std::vector<double> &actual)
{
     double sum = 0.0;
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double x = actual[i] - expected[i];
          sum += std::log(std::cosh(x));
     }
     return sum / expected.size();
}

std::vector<double> CostFunction::logCoshDerivative(const std::vector<double> &expected, const std::vector<double> &actual)
{
     std::vector<double> grad(expected.size());
     for (size_t i = 0; i < expected.size(); ++i)
     {
          double x = actual[i] - expected[i];
          grad[i] = std::tanh(x);
     }
     return grad;
}
