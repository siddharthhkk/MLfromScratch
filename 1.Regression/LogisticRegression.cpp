#include <bits/stdc++.h>
using namespace std;
const int n = 2, m = 6;

double sigmoid(double z){
    return 1.0 / (1.0 + exp(-z));
}

void z_normalize(double x[n][m], double x_scaled[n][m]){
    for(int j = 0; j < n; j++){
        double mean = 0, sd = 0;
        for(int i = 0; i < m; i++){
            mean += x[j][i];
        }
        mean /= m;
        for(int i = 0; i < m; i++){
            sd += pow(x[j][i] - mean, 2);
        }
        sd = sqrt(sd / m);
        for(int i = 0; i < m; i++){
            x_scaled[j][i] = (x[j][i] - mean) / sd;
        }
    }
}
double compute_cost(double x[n][m], double w[n], double b, double y[m]){
    double j_wb = 0;
    for(int i = 0; i < m; i++){
        double z = 0;
        for(int j = 0; j < n; j++){
            z += w[j] * x[j][i];
        }
        z += b;
        double y_cap = sigmoid(z);
        j_wb += y[i] * log(y_cap) + (1 - y[i]) * log(1 - y_cap);
    }
    j_wb *= -(1.0 / m);
    return j_wb;
}

void compute_gradient(double x[n][m], double w[n], double b, double y[m], double dj_dw[n], double &dj_db){
    for(int j = 0; j < n; j++){
        dj_dw[j] = 0;
    }
    dj_db = 0;
    for(int i = 0; i < m; i++){
        double z = 0;
        for(int j = 0; j < n; j++){
            z += w[j] * x[j][i];
        }
        z += b;
        double y_cap = sigmoid(z);
        for(int j = 0; j < n; j++){
            dj_dw[j] += (y_cap - y[i]) * x[j][i];
        }
        dj_db += (y_cap - y[i]);
    }
    dj_db *= (1.0 / m);
    for(int j = 0; j < n; j++){
        dj_dw[j] *= (1.0 / m);
    }
}

void gradient_descent(double x[n][m], double y[m], double w[n], double &b, double alpha, int iterations){
    double dj_dw[n], dj_db;
    vector<double> history(iterations);
    double previous_cost = 0.0;
    int actual_iters = 0;
    for(int i = 0; i < iterations; i++){
        compute_gradient(x, w, b, y, dj_dw, dj_db);
        for(int j = 0; j < n; j++){
            w[j] -= alpha * dj_dw[j];
        }
        b -= alpha * dj_db;
        history[i] = compute_cost(x, w, b, y);
        actual_iters++;
        if (i == 0 || (i + 1) % 100 == 0) {
            cout << "Iteration " << i + 1 << ": Cost = " << history[i] << endl;
        }
        // Early stopping if cost stops changing significantly
        if (i > 0) {
            double cost_change = std::abs(previous_cost - history[i]);
            if (cost_change < 1e-7) {
                cout << "Converged early at iteration " << i + 1 << "!" << endl;
                break;
            }
        }
        previous_cost = history[i];
    }
}

int main(){
    double x[n][m] = {
        {10, 2, 8, 3, 9, 1},
        {5, 1, 4, 2, 5, 2}
    };
    double y[m] = {1, 0, 1, 0, 1, 0};
    double x_scaled[n][m];
    z_normalize(x, x_scaled);
    double w[n] = {0, 0};
    double b = 0.0;
    gradient_descent(x_scaled, y, w, b, 0.15, 5000);
    cout << "Final weights: ";
    for(int j = 0; j < n; j++){
        cout << w[j] << " ";
    }
    cout << "\nFinal bias: " << b << endl;
    cout << "Final cost: " << compute_cost(x_scaled, w, b, y) << endl;
}