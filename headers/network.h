#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <string>

struct LayerOutput {
    std::vector<std::vector<double>> Z;
    std::vector<std::vector<double>> A;
};

double relu(double z);
std::vector<double> softmax(std::vector<double> z, int units);
LayerOutput Dense(std::vector<std::vector<double>> &a, int units, std::string act, std::vector<std::vector<double>> &W, std::vector<double> &b);
std::vector<std::vector<double>> predict(std::vector<std::vector<double>>& X_new, std::vector<std::vector<vector<double>>>& W, std::vector<std::vector<double>>& b);

#endif