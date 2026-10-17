#include <bits/stdc++.h>
#include <fstream>
using namespace std;
#include "NeuralNetworks.h"

// g++ Predict.cpp NeuralNetworks.cpp -o Predict.exe && ./Predict.exe
bool load_weights(const string& filename, vector<vector<vector<double>>>& W, vector<vector<double>>& b) {
    ifstream inFile(filename);
    size_t num_layers; inFile >> num_layers; W.resize(num_layers);
    for (size_t l = 0; l < num_layers; l++) {
        size_t rows, cols; 
        inFile >> rows >> cols;
        if(rows > 1000 || cols > 1000) {
            cout << "CRITICAL ERROR: Detected corrupt dimensions! Aborting allocation." << endl;
            return false;
        }
        W[l].assign(rows, vector<double>(cols));
        for (size_t j = 0; j < rows; j++)
            for (size_t k = 0; k < cols; k++) inFile >> W[l][j][k];
    }
    size_t num_bias_layers; 
    inFile >> num_bias_layers; 
    b.resize(num_bias_layers);
    for (size_t l = 0; l < num_bias_layers; l++) {
        size_t rows; 
        if (!(inFile >> rows)) return false;
        b[l].resize(rows);
        for (size_t j = 0; j < rows; j++) {
            if (!(inFile >> b[l][j])) return false;
        }
    }
    inFile.close();
    cout << "Model configurations successfully loaded into memory!" << endl;
    return true;
}

int main(){
    vector<vector<vector<double>>> W;
    vector<vector<double>> b;
    if (!load_weights("weights", W, b)) { 
        cout << "Execution stopped cleanly to prevent vector crash." << endl;
        return 0; 
    }
    vector<vector<double>> x_test = {{-0.80,  0.20}, { 0.80, -0.80}};
    vector<vector<double>> test_predictions = predict(x_test, W, b);
     cout << "\n--- Instant Prediction Results ---" << endl;
    for (int i = 0; i < 2; i++) {
        int predicted_class = 0;
        double max_prob = test_predictions[0][i];
        for (int j = 1; j < 3; j++) {
            if (test_predictions[j][i] > max_prob) {
                max_prob = test_predictions[j][i];
                predicted_class = j;
            }
        }
        cout << "Sample " << i << " | Predicted Class: " << predicted_class 
             << " | Confidence: " << fixed << setprecision(2) << max_prob * 100.0 << "%" << endl;
    }
}