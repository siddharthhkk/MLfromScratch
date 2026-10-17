#include <bits/stdc++.h>
#include <fstream>
using namespace std;
#include "NeuralNetworks.h"

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
vector<vector<double>> sigmoid(vector<vector<double>>& z){
    int rows = z.size();
    int cols = z[0].size();
    vector<vector<double>> out(rows, vector<double>(cols));
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            out[i][j] = 1.0 / (1.0 + exp(-z[i][j]));
        }
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
// ya i cried! 4 laps 55 minutes each!! and it works 