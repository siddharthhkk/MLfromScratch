#pragma once
struct LayerOutput {
    std::vector<std::vector<double>> Z;
    std::vector<std::vector<double>> A;
};
double relu(double z);
vector<double> softmax(vector<double> z, int units);
LayerOutput Dense(vector<vector<double>> &a, int units, string act, vector<vector<double>> &W, vector<double> &b);
void save_weights(const string& filename, const vector<vector<vector<double>>>& W, const vector<vector<double>>& b);
double categorical_cross_entropy(vector<double> &y, vector<vector<double>> &a_out);
vector<vector<double>> sigmoid(vector<vector<double>>& z);
vector<vector<double>> predict(vector<vector<double>>& X_new, vector<vector<vector<double>>>& W, vector<vector<double>>& b);
void backward(vector<vector<double>>& X, vector<double>& y, vector<vector<double>>& Z1, vector<vector<double>>& A1,
    vector<vector<double>>& Z2, vector<vector<double>>& A2, vector<vector<double>>& A3, vector<vector<double>>& W1, 
    vector<double>& b1, vector<vector<double>>& W2, vector<double>& b2, vector<vector<double>>& W3, vector<double>& b3, 
    double alpha);