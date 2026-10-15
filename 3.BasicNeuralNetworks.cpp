#include <bits/stdc++.h>
#include <fstream>
using namespace std;
#include "headers/network.h"

double relu(double z){
    return max(0.0, z);
}
vector<double> softmax(vector<double> z, int units){
    double sum = 0;
    vector<double> out(units);
    for(int i = 0; i < z.size(); i++){
        sum += exp(z[i]);
    }
    for(int i = 0; i < units; i++){
        out[i] = exp(z[i]) / sum;
    }
    return out;
}

double categorical_cross_entropy(vector<double> &y, vector<vector<double>> &a_out){
    double loss = 0;
    for(int i = 0; i < a_out.size(); i++){
        int target = static_cast<int>(y[i]);
        double predicted = a_out[target][i];
        predicted = max(predicted, 1e-15); // Avoid log(0)  
        loss += -log(predicted);
    }
    return loss / a_out.size();
}
double relu_derivative(double z){
    return z > 0 ? 1 : 0;
}
// shit's Traumatic
void backward(vector<vector<double>>& X, vector<double>& y, vector<vector<double>>& Z1, vector<vector<double>>& A1,
    vector<vector<double>>& Z2, vector<vector<double>>& A2, vector<vector<double>>& A3, vector<vector<double>>& W1, 
    vector<double>& b1, vector<vector<double>>& W2, vector<double>& b2, vector<vector<double>>& W3, vector<double>& b3, 
    double alpha){
    int m = y.size();
    // dZ3 shape: (3 x 12) -> dZ3 = A3 - Y_one_hot
    vector<vector<double>> dZ3(3, vector<double>(m));
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < m; i++) {
            double y_one_hot = (static_cast<int>(y[i]) == j) ? 1.0 : 0.0;
            dZ3[j][i] = A3[j][i] - y_one_hot;
        }
    }

    // dW3 shape: (3 x 15) -> dW3 = (1/m) * dZ3 * A2^T
    vector<vector<double>> dW3(3, vector<double>(15, 0.0));
    vector<double> db3(3, 0.0);
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 15; k++) {
            for (int i = 0; i < m; i++) {
                dW3[j][k] += dZ3[j][i] * A2[k][i];
            }
            dW3[j][k] /= m;
        }
        for (int i = 0; i < m; i++) {
            db3[j] += dZ3[j][i];
        }
        db3[j] /= m;
    }

    // dA2 shape: (15 x 12) -> dA2 = W3^T * dZ3
    vector<vector<double>> dA2(15, vector<double>(m, 0.0));
    for (int j = 0; j < 15; j++) {
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < 3; k++) {
                dA2[j][i] += W3[k][j] * dZ3[k][i];
            }
        }
    }

    // dZ2 shape: (15 x 12) -> dZ2 = dA2 * relu_derivative(Z2)
    vector<vector<double>> dZ2(15, vector<double>(m));
    for (int j = 0; j < 15; j++) {
        for (int i = 0; i < m; i++) {
            dZ2[j][i] = dA2[j][i] * relu_derivative(Z2[j][i]);
        }
    }

    // dW2 shape: (15 x 25) -> dW2 = (1/m) * dZ2 * A1^T
    vector<vector<double>> dW2(15, vector<double>(25, 0.0));
    vector<double> db2(15, 0.0);
    for (int j = 0; j < 15; j++) {
        for (int k = 0; k < 25; k++) {
            for (int i = 0; i < m; i++) {
                dW2[j][k] += dZ2[j][i] * A1[k][i];
            }
            dW2[j][k] /= m;
        }
        for (int i = 0; i < m; i++) {
            db2[j] += dZ2[j][i];
        }
        db2[j] /= m;
    }

    // dA1 shape: (25 x 12) -> dA1 = W2^T * dZ2
    vector<vector<double>> dA1(25, vector<double>(m, 0.0));
    for (int j = 0; j < 25; j++) {
        for (int i = 0; i < m; i++) {
            for (int k = 0; k < 15; k++) {
                dA1[j][i] += W2[k][j] * dZ2[k][i];
            }
        }
    }

    vector<vector<double>> dZ1(25, vector<double>(m));
    for (int j = 0; j < 25; j++) {
        for (int i = 0; i < m; i++) {
            dZ1[j][i] = dA1[j][i] * relu_derivative(Z1[j][i]);
        }
    }

    // dW1 shape: (25 x 2) -> dW1 = (1/m) * dZ1 * X^T
    vector<vector<double>> dW1(25, vector<double>(2, 0.0));
    vector<double> db1_grad(25, 0.0);
    for (int j = 0; j < 25; j++) {
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < m; i++) {
                dW1[j][k] += dZ1[j][i] * X[k][i];
            }
            dW1[j][k] /= m;
        }
        for (int i = 0; i < m; i++) {
            db1_grad[j] += dZ1[j][i];
        }
        db1_grad[j] /= m;
    }
    // GRADIENT DESCENT PARAMETER UPDATE: Update Layer 3
    for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 15; k++) W3[j][k] -= alpha * dW3[j][k];
        b3[j] -= alpha * db3[j];
    }
    // Update Layer 2
    for (int j = 0; j < 15; j++) {
        for (int k = 0; k < 25; k++) W2[j][k] -= alpha * dW2[j][k];
        b2[j] -= alpha * db2[j];
    }
    // Update Layer 1
    for (int j = 0; j < 25; j++) {
        for (int k = 0; k < 2; k++) W1[j][k] -= alpha * dW1[j][k];
        b1[j] -= alpha * db1_grad[j];
    }
}

LayerOutput Dense(vector<vector<double>> &a, int units, string act, vector<vector<double>> &W, vector<double> &b){
    int num_samples = a[0].size(); // 12 samples (columns)
    int num_features = a.size();   // 2 features (rows)
    vector<vector<double>> z_out(units, vector<double>(num_samples));
    vector<vector<double>> a_out(units, vector<double>(num_samples));
    for(int i = 0; i < num_samples; i++){
        for(int j = 0; j < units; j++){
            double z_sum = 0;
            for(int k = 0; k < num_features; k++){
                z_sum += a[k][i] * W[j][k];
            }z_out[j][i] = z_sum + b[j];
        }
        if(act == "relu"){
            for(int j = 0; j < units; j++){
                a_out[j][i] = relu(z_out[j][i]);    
            }
        }
        else if(act == "softmax"){
            for (int i = 0; i < num_samples; i++) {
                vector<double> z_sample(units);
                for (int j = 0; j < units; j++) {
                    z_sample[j] = z_out[j][i];
                }
                vector<double> a_sample = softmax(z_sample, units);
                for (int j = 0; j < units; j++) {
                    a_out[j][i] = a_sample[j];
                }
            }
        }
    }
    return {z_out, a_out};
}


vector<vector<double>> predict(vector<vector<double>>& X_new, vector<vector<vector<double>>>& W, vector<vector<double>>& b){
    // running new data through our trained weights and biases!
    LayerOutput l1 = Dense(X_new, 25, "relu", W[0], b[0]);
    LayerOutput l2 = Dense(l1.A, 15, "relu", W[1], b[1]);
    LayerOutput l3 = Dense(l2.A, 3, "softmax", W[2], b[2]);
    return l3.A;
}

void save_weights(const string& filename, const vector<vector<vector<double>>>& W, const vector<vector<double>>& b) {
    ofstream outFile(filename);
    outFile << W.size() << "\n";
    for (size_t l = 0; l < W.size(); l++) {
        outFile << W[l].size() << " " << W[l][0].size() << "\n";
        for (size_t j = 0; j < W[l].size(); j++) {
            for (size_t k = 0; k < W[l][j].size(); k++) outFile << W[l][j][k] << " ";
            outFile << "\n";
        }
    }
    outFile << b.size() << "\n";
    for (size_t l = 0; l < b.size(); l++) {
        outFile << b[l].size() << "\n";
        for (size_t j = 0; j < b[l].size(); j++) outFile << b[l][j] << " ";
        outFile << "\n";
    }
    outFile.close();
    cout << "Weights saved to " << filename << endl;
}

#ifndef PREDICT_MODE
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
    save_weights("3.Weights.txt", W, b);


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
#endif
// ya i cried! 4 laps 55 minutes each!! and it works 