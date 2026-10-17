#include <bits/stdc++.h>
#include <fstream>
using namespace std;
#include "NeuralNetworks.h"

// g++ Train.cpp NeuralNetworks.cpp -o Train.exe && ./Train.exe
int main(){
    vector<vector<double>> x = {{-0.85, -0.71, -0.93, -0.62, 0.12, 0.28, -0.05, 0.33, 0.75, 0.89, 0.61, 0.82},
                                {0.72, 0.88, 0.55, 0.79, -0.91, -0.75, -0.88, -0.61, 0.68, 0.52, 0.83, 0.71}};
    vector<double> y = {0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2};
    vector<int> layer_sizes = {2, 25, 15, 3};
    int total_layers = layer_sizes.size() - 1;
    vector<vector<vector<double>>> W(total_layers);
    vector<vector<double>> b(total_layers);
    mt19937 gen(42);
    uniform_real_distribution<double> dist(-0.1, 0.1);
    for(int i = 0; i < total_layers; i++){
        int rows = layer_sizes[i + 1];
        int cols = layer_sizes[i]; // features in current layer
        W[i] = vector<vector<double>>(rows, vector<double>(cols));
        b[i] = vector<double>(rows, 0.0);
        for(int j = 0; j < rows; j++){
            for(int k = 0; k < cols; k++){
                W[i][j][k] = dist(gen);
            }
            b[i][j] = dist(gen);
        }
    }
    int epochs = 5000;
    double alpha = 0.02;
    cout << "Starting Training Loops..." << endl;
    vector<vector<double>> A3;
    for(int epoch = 0; epoch < epochs; epoch++){
        LayerOutput l_1 = Dense(x, 25, "relu", W[0], b[0]);
        vector<vector<double>> Z1 = l_1.Z;
        vector<vector<double>> A1 = l_1.A;
        
        LayerOutput l_2 = Dense(A1, 15, "relu", W[1], b[1]);
        vector<vector<double>> Z2 = l_2.Z;
        vector<vector<double>> A2 = l_2.A;

        LayerOutput l_3 = Dense(A2, 3, "softmax", W[2], b[2]);
        A3 = l_3.A;
        if(epoch % 100 == 0){
            double cost = categorical_cross_entropy(y, A3);
            cout << "Epoch: " << epoch << ", Loss: " << cost << endl;
        }
        backward(x, y, Z1, A1, Z2, A2, A3, W[0], b[0], W[1], b[1], W[2], b[2], alpha);
    }
    save_weights("weights", W, b);


    int correct_predictions = 0;
    int total_samples = y.size();

    cout << "\n--- Final Evaluation ---" << endl;
    for (int i = 0; i < total_samples; i++) {
        // Find the index with the highest probability (argmax) for sample i
        int predicted_class = 0;
        double max_prob = A3[0][i];
        
        for (int j = 1; j < 3; j++) {
            if (A3[j][i] > max_prob) {
                max_prob = A3[j][i];
                predicted_class = j;
            }
        }
        int true_class = static_cast<int>(y[i]);
        if (predicted_class == true_class) {
            correct_predictions++;
        }
        cout << "Sample " << i << " | Predicted: " << predicted_class 
             << " | True: " << true_class 
             << " | Prob: " << fixed << setprecision(2) << max_prob << endl;
    }

    double final_accuracy = (static_cast<double>(correct_predictions) / total_samples) * 100.0;
    cout << "\nFinal Accuracy: " << final_accuracy << "% (" 
    << correct_predictions << "/" << total_samples << " correct)" << endl;
}