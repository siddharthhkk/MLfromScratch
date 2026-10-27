#include <bits/stdc++.h>
using namespace std;

class AdamOptimizer {
private:
    double alpha;   
    double beta1;   // Exponential decay rate for the 1st moment
    double beta2;   // Exponential decay rate for the 2nd moment
    double epsilon; 
    int t;          

    vector<double> m; // First moment vector
    vector<double> v; // Second moment vector
public:
    // Initialize with standard default hyperparameters
    AdamOptimizer(size_t num_params, double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : alpha(lr), beta1(b1), beta2(b2), epsilon(eps), t(0) {
        m.resize(num_params, 0.0);
        v.resize(num_params, 0.0);
    }

    // Performs a single optimization step given parameters and their gradients
    void step(std::vector<double>& params, const std::vector<double>& grads) {
        if (params.size() != grads.size() || params.size() != m.size()) {
            throw std::invalid_argument("Vector sizes do not match optimizer parameter count.");
        }
        t++;
        double bias_correction1 = 1.0 - std::pow(beta1, t);
        double bias_correction2 = 1.0 - std::pow(beta2, t);
        for (size_t i = 0; i < params.size(); ++i) {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];

            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];

            // Compute bias-corrected first moment estimate
            double m_hat = m[i] / bias_correction1;
            // Compute bias-corrected second raw moment estimate
            double v_hat = v[i] / bias_correction2;
            params[i] -= (alpha / (std::sqrt(v_hat) + epsilon)) * m_hat;
        }
    }
    void reset() {
        t = 0;
        fill(m.begin(), m.end(), 0.0);
        fill(v.begin(), v.end(), 0.0);
    }
};

int main() {
    vector<double> parameters = {5.0, 5.0};
    
    AdamOptimizer optimizer(parameters.size(), /*lr=*/0.1);

    cout << "Starting optimization...\n";
    for (int epoch = 1; epoch <= 100; ++epoch) {
        // Analytical gradients of f(x, y) = x^2 + 3y^2
        // df/dx = 2x,  df/dy = 6y
        std::vector<double> gradients = {
            2.0 * parameters[0],
            6.0 * parameters[1]
        };

        optimizer.step(parameters, gradients);

        if (epoch % 10 == 0) {
            double loss = std::pow(parameters[0], 2) + 3.0 * std::pow(parameters[1], 2);
            std::cout << "Epoch " << epoch << " | Params: (" << parameters[0] << ", " << parameters[1] << ") | Loss: " << loss << "\n";
        }
    }
}
