#include <bits/stdc++.h>
using namespace std;

vector<double> estimate_mean(vector<vector<double>>& X){
    int m = X.size();
    int n = X[0].size();
    vector<double> mean(n, 0.0);
    for(int j = 0; j < n; j++){
        double sum = 0.0;
        for(int i = 0; i < m; i++){
            sum += X[i][j];
        }
        mean[j] = sum / m;
    }
    return mean;
}
vector<double> estimate_variance(vector<vector<double>> &X, vector<double> &mean){
    int m = X.size();
    int n = X[0].size();
    vector<double> variance(n, 0.0);
    for(int j = 0; j < n; j++){
        double sum = 0.0;
        for(int i = 0; i < m; i++){
            sum += pow((X[i][j] - mean[j]), 2);
        }
        variance[j] = sum / m;
    }
    return variance;
}
vector<double> gaussian_probability(vector<vector<double>> &X, vector<double> &means, vector<double> &variance){
    int m = X.size();
    int n = X[0].size();
    vector<double> dataset_probabilities(m, 1.0);
    for(int i = 0; i < m; i++){
        double p = 1.0;
        for(int j = 0; j < n; j++){
            double mean = means[j];
            double var = variance[j];

            if (var == 0.0) var = 1e-9;
            double denominator = sqrt(2 * M_PI * var);
            double exponent = exp(-pow((X[i][j] - mean), 2) / (2 * var));
            p *= (1 / denominator) * exponent;
        }
        dataset_probabilities[i] = p;
    }
    return dataset_probabilities;
}
int main() {
    vector<vector<double>> X = {
        // Normal Data Points
        {10.1, 9.9}, {9.8, 10.2}, {10.5, 9.5}, {9.9, 10.1}, {10.2, 10.3}, {9.5, 9.7}, {10.0, 10.0}, {10.3, 9.8},
        // Anomalous Data Points
        {10.2, 2.1}, {22.5, 21.0}, {1.1, 15.6}};
    
    vector<vector<double>> X_normal(X.begin(), X.begin() + 8); // First 8 points are normal
    vector<double> means = estimate_mean(X_normal);
    vector<double> variance = estimate_variance(X_normal, means);
    vector<double> probabilities = gaussian_probability(X, means, variance);
    double epsilon = 0.01;
    cout << "Anomalies detected at indices: ";
    for(int i = 0; i < probabilities.size(); i++){
        cout << "Point " << i << " -> Probability: " << probabilities[i];
        if(probabilities[i] < epsilon) cout << " [ANOMALY DETECTED] 🚨";
        else cout << " [Normal]";
        cout << "\n";
    }
}