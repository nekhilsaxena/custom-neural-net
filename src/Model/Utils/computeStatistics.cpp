#include <vector>
#include <algorithm>
#include <unordered_map>
#include <cmath>
#include <stdexcept>
#include <limits>
#include "Utils.h"

float computeMedian(const std::vector<float> &vec)
{
     size_t size = vec.size();
     if (size == 0)
          return 0.0f;
     if (size % 2 == 0)
          return (vec[size / 2 - 1] + vec[size / 2]) / 2.0f;
     else
          return vec[size / 2];
}

StatsSummary computeStatistics(const std::vector<float> &data)
{
     if (data.empty())
          throw std::runtime_error("Cannot compute statistics on an empty dataset.");

     std::vector<float> sorted = data;
     std::sort(sorted.begin(), sorted.end());
     int n = static_cast<int>(sorted.size());

     // Mean
     float sum = 0.0f;
     for (float val : sorted)
          sum += val;
     float mean = sum / n;

     // Median
     float median = computeMedian(sorted);

     // Q1 and Q3
     std::vector<float> lower, upper;
     int mid = n / 2;

     if (n >= 4)
     {
          lower = std::vector<float>(sorted.begin(), sorted.begin() + mid);
          upper = (n % 2 == 0)
                      ? std::vector<float>(sorted.begin() + mid, sorted.end())
                      : std::vector<float>(sorted.begin() + mid + 1, sorted.end());
     }
     else
     {
          lower = sorted;
          upper = sorted;
     }

     float q1 = computeMedian(lower);
     float q3 = computeMedian(upper);
     float iqr = q3 - q1;

     // Outlier bounds
     float lowerBound = q1 - 1.5f * iqr;
     float upperBound = q3 + 1.5f * iqr;

     // Min/Max excluding outliers
     float minVal = std::numeric_limits<float>::max();
     float maxVal = std::numeric_limits<float>::lowest();
     for (float val : sorted)
     {
          if (val >= lowerBound && val <= upperBound)
          {
               if (val < minVal)
                    minVal = val;
               if (val > maxVal)
                    maxVal = val;
          }
     }

     if (minVal == std::numeric_limits<float>::max())
          minVal = sorted.front(); // fallback
     if (maxVal == std::numeric_limits<float>::lowest())
          maxVal = sorted.back(); // fallback

     // Mode
     std::unordered_map<float, int> freq;
     for (float val : sorted)
          freq[val]++;

     float mode = sorted[0];
     int maxFreq = 0;
     for (const auto &[val, count] : freq)
     {
          if (count > maxFreq)
          {
               maxFreq = count;
               mode = val;
          }
     }

     return StatsSummary{mean, median, mode, q1, q3, minVal, maxVal};
}
