#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

std::vector<std::vector<std::string>> readRawCSV(const std::string &filename)
{
     std::ifstream file(filename);
     if (!file)
          throw std::runtime_error("Failed to open CSV file: " + filename);

     std::vector<std::vector<std::string>> data;
     std::string line;

     while (std::getline(file, line))
     {
          std::stringstream lineStream(line);
          std::vector<std::string> row;
          std::string cell;

          while (std::getline(lineStream, cell, ','))
          {
               row.push_back(cell);
          }

          data.push_back(row);
     }

     if (data.empty())
          throw std::runtime_error("CSV is empty or malformed.");

     return data;
}
