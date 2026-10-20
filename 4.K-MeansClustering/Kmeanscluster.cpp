#include <bits/stdc++.h>
#include <cstdlib>
#include <ctime> 
using namespace std;

double getDistance(vector<double>& point1, const vector<double>& point2) {
    double distance = 0.0;
    for (size_t i = 0; i < point1.size(); i++) {
        distance += pow(point1[i] - point2[i], 2);
    }
    return distance;
}

vector<vector<double>> initializeCentroids(vector<vector<double>>& X, int K) {
    vector<vector<double>> centroids;
    srand(time(0)); 
    int first_idx = rand() % X.size();
    centroids.push_back(X[first_idx]);
    
    for (int i = 1; i < K; i++) {
        vector<double> distances_squared(X.size(), 0.0);
        double total_sum = 0.0;
        for (size_t j = 0; j < X.size(); j++) {
            double min_dist = numeric_limits<double>::max();
            for (const auto& c : centroids) {
                double dist = getDistance(X[j], c);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            distances_squared[j] = min_dist;
            total_sum += min_dist;
        }
        double target = ((double)rand() / RAND_MAX) * total_sum;
        double current_sum = 0.0;
        int selected_idx = 0;
        
        for (size_t j = 0; j < X.size(); j++) {
            current_sum += distances_squared[j];
            if (current_sum >= target) {
                selected_idx = j;
                break;
            }
        }
        centroids.push_back(X[selected_idx]);
    }
    return centroids;
}


vector<int> assignClusters(vector<vector<double>>& X, vector<vector<double>>& centroids) {
    vector<int> assignments(X.size());

    for (size_t i = 0; i < X.size(); i++) {
        double min_dist = INT_MAX;
        int closest_centroid = -1;

        for (size_t j = 0; j < centroids.size(); j++) {
            double dist = getDistance(X[i], centroids[j]);
            if (dist < min_dist) {
                min_dist = dist;
                closest_centroid = j;
            }
        }
        assignments[i] = closest_centroid; 
    }
    return assignments;
}
vector<vector<double>> updateCentroids(const vector<vector<double>>& X, const vector<int>& assignments, int K) {
    int dimensions = X[0].size();
    vector<vector<double>> new_centroids(K, vector<double>(dimensions, 0.0));
    vector<int> counts(K, 0);

    for (size_t i = 0; i < X.size(); i++) {
        int cluster_id = assignments[i];
        counts[cluster_id]++;
        for (int d = 0; d < dimensions; d++) {
            new_centroids[cluster_id][d] += X[i][d];
        }
    }

    for (int i = 0; i < K; i++) {
        if (counts[i] > 0) {
            for (int d = 0; d < dimensions; d++) {
                new_centroids[i][d] /= counts[i];
            }
        } else new_centroids[i] = X[rand() % X.size()]; 
    }
    return new_centroids;
}


int main(){
    vector<vector<double>> X = {{1.0, 1.0}, {1.5, 2.0}, {2.0, 1.5}, {1.2, 1.8},
    {8.0, 8.0}, {8.5, 9.0}, {9.0, 8.5}, {7.8, 8.8}, {8.0, 1.0}, {8.5, 1.5}, {9.0, 1.2}, {7.9, 1.1}};
    int K = 3;
    int max_iterations = 100;
    vector<vector<double>> centroids = initializeCentroids(X, K);
    vector<int> assignments;

    for (int i = 0; i < max_iterations; i++) {
        assignments = assignClusters(X, centroids);
        vector<vector<double>> new_centroids = updateCentroids(X, assignments, K);
        
        bool converged = true;
        for (int i = 0; i < K; i++) {
            if (getDistance(centroids[i], new_centroids[i]) > 0.0001) {
                converged = false;
                break;
            }
        }
        centroids = new_centroids;
        if (converged) {
            cout << "Converged early at iteration: " << i + 1 << "\n\n";
            break;
        }
    }
    for (int i = 0; i < K; i++) {
        cout << "--- Cluster " << i << " --- (Centroid: [" << centroids[i][0] << ", " << centroids[i][1] << "])\n";
        for (size_t j = 0; j < X.size(); j++) {
            if (assignments[j] == i) {
                cout << "  Point: [" << X[j][0] << ", " << X[j][1] << "]\n";
            }
        }
    }
}