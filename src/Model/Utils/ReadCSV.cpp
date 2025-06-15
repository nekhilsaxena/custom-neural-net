#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>

std::vector<std::vector<double>> readCSV(const std::string &filename)
{
     std::vector<std::vector<double>> data;
     std::ifstream file(filename);
     if (!file)
     {
          throw std::runtime_error("Failed to open CSV file: " + filename);
     }

     std::string line;

     if (!std::getline(file, line))
     {
          throw std::runtime_error("CSV file is empty or missing header");
     }

     while (std::getline(file, line))
     {
          std::vector<double> row;
          std::stringstream ss(line);
          std::string value;

          while (std::getline(ss, value, ','))
          {
               // Trim leading/trailing whitespace
               value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch)
                                                       { return !std::isspace(ch); }));
               value.erase(std::find_if(value.rbegin(), value.rend(), [](unsigned char ch)
                                        { return !std::isspace(ch); })
                               .base(),
                           value.end());

               if (value.empty())
               {
                    throw std::invalid_argument("Empty field encountered in CSV");
               }

               try
               {
                    row.push_back(std::stod(value));
               }
               catch (const std::invalid_argument &e)
               {
                    throw std::invalid_argument("Invalid numeric value in CSV: \"" + value + "\"");
               }
          }

          if (!row.empty())
          {
               data.push_back(row);
          }
     }

     return data;
}