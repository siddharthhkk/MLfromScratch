#include <bits/stdc++.h>
using namespace std;

#include <vector>
#include <cmath>
#include <iostream>

class RMSprop {
public:
    double lr;       // alpha
    double gamma;    // Decay rate / rho 
    double epsilon;
    vector<double> cache;
    RMSprop(size_t size, double learning_rate = 0.01, double decay = 0.9, double eps = 1e-8): 
    lr(learning_rate), gamma(decay), epsilon(eps), cache(size, 0.0, allocator<double>()) {}

    void update(vector<double>& params, const vector<double>& grads) {
        for (size_t i = 0; i < params.size(); ++i) {
            // Update running average of squared gradients: v = gamma * v + (1 - gamma) * g^2
            cache[i] = gamma * cache[i] + (1.0 - gamma) * grads[i] * grads[i];
            
            // Scale parameter: theta = theta - (lr / (sqrt(v) + epsilon)) * g
            params[i] -= (lr / (std::sqrt(cache[i]) + epsilon)) * grads[i];
        }
    }
};
