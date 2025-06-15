#ifndef OPERATIONS_H
#define OPERATIONS_H

#include <vector>
#include <string>

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

std::vector<std::vector<double>> readCSV(const std::string &filename);
std::vector<std::vector<std::string>> readRawCSV(const std::string &filename);
double computeMSE(const std::vector<double> &expected, const std::vector<double> &actual);
StatsSummary computeStatistics(const std::vector<float> &data);

#endif
