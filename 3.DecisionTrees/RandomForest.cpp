#include <bits/stdc++.h>
#include <random>
using namespace std;
#include "DecisionTree.h"

// g++ RandomForest.cpp DecisionTree.cpp -o RandomForest.exe && ./RandomForest.exe
double find_best_split_rf(vector<vector<double>> &X, vector<int> &y, int &best_feature_index, double &best_threshold, int max_features){
    double max_gain = -1;
    best_feature_index = -1;
    best_threshold = -1;
    int num_samples = X.size();
    int num_features = X[0].size();
    vector<int> feature_indices(num_features);
    for(int i = 0; i < num_features; i++) feature_indices[i] = i;
    random_device rd;
    mt19937 g(rd());
    shuffle(feature_indices.begin(), feature_indices.end(), g);

    int features_to_evaluate = min(max_features, num_features);
    for(int j = 0; j < features_to_evaluate; j++){
        int jj = feature_indices[j];
        for(int i = 0; i < num_samples; i++){
            double threshold = X[i][jj];
            double gain = information_gain(X, y, jj, threshold);
            if(gain > max_gain){
                max_gain = gain;
                best_feature_index = jj;
                best_threshold = threshold;
            }
        }
    }
    return max_gain;
}

void bootstrap_sample(vector<vector<double>> &X, vector<int> &y, vector<vector<double>> &X_boot, vector<int> &y_boot){
    int n_samples = X.size();
    X_boot.clear();
    y_boot.clear();
    X_boot.reserve(n_samples);
    y_boot.reserve(n_samples);

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, n_samples - 1);
    for(int i = 0; i < n_samples; i++){
        int idx = dis(gen);
        X_boot.push_back(X[idx]);
        y_boot.push_back(y[idx]);
    }
}

shared_ptr<Node> grow_tree_rf(vector<vector<double>> &X, vector<int> &y, int depth, int max_depth, int max_features) {
    shared_ptr<Node> node = make_shared<Node>();
    int count1 = count(y.begin(), y.end(), 1);
    int majority_class = (count1 >= ((int)y.size() - count1)) ? 1 : 0;

    if (depth >= max_depth || entropy(y) == 0 || y.empty()) {
        node->is_leaf = true;
        node->prediction = majority_class;
        return node;
    }
    int best_feat;
    double best_thresh;
    double max_gain = find_best_split_rf(X, y, best_feat, best_thresh, max_features);
    if (max_gain <= 1e-6) {
        node->is_leaf = true;
        node->prediction = majority_class;
        return node;
    }
    node->feature_index = best_feat;
    node->threshold = best_thresh;
    vector<vector<double>> left_X, right_X;
    vector<int> left_y, right_y;
    
    for (size_t i = 0; i < X.size(); i++) {
        if (X[i][best_feat] <= best_thresh) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        } else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
        }
    }
    
    node->left = grow_tree_rf(left_X, left_y, depth + 1, max_depth, max_features);
    node->right = grow_tree_rf(right_X, right_y, depth + 1, max_depth, max_features);
    return node;
}

vector<shared_ptr<Node>> build_random_forest(vector<vector<double>> &X, vector<int> &y, int num_trees, int max_depth, int max_features){
    vector<shared_ptr<Node>> forest;
    forest.reserve(num_trees);
    if (max_features <= 0 && !X.empty()) {
        max_features = max(1, (int)sqrt(X[0].size()));
    }
    for (int i = 0; i < num_trees; i++) {
        vector<vector<double>> X_boot;
        vector<int> y_boot;
    
        bootstrap_sample(X, y, X_boot, y_boot);
        shared_ptr<Node> tree_root = grow_tree_rf(X_boot, y_boot, 0, max_depth, max_features);
        forest.push_back(tree_root);
    }
    return forest;
}

int predict_random_forest(vector<shared_ptr<Node>> &forest, vector<double> &x) {
    int votes_for_0 = 0;
    int votes_for_1 = 0;
    for (auto &tree_root : forest) {
        int vote = predict(tree_root, x); 
        if (vote == 1) votes_for_1++;
        else votes_for_0++;
    }
    return (votes_for_1 >= votes_for_0) ? 1 : 0;
}

int main() {
    vector<vector<double>> X_train = {{1.4, 0.2}, {1.3, 0.2}, {1.5, 0.2}, {1.4, 0.3}, {1.7, 0.4},
                                      {4.7, 1.4}, {4.5, 1.5}, {4.9, 1.5}, {4.0, 1.3}, {4.6, 1.3}};
    vector<int> y_train = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    int num_trees = 15;
    int max_depth = 3;
    int max_features = 1;

    cout << "Training single-file Random Forest..." << endl;
    vector<shared_ptr<Node>> my_forest = build_random_forest(X_train, y_train, num_trees, max_depth, max_features);
    cout << "Forest complete with " << my_forest.size() << " trees." << endl << endl;
    vector<double> test_setosa = {1.4, 0.2};
    vector<double> test_versicolor = {4.2, 1.4}; 
    int pred_setosa = predict_random_forest(my_forest, test_setosa);
    int pred_versicolor = predict_random_forest(my_forest, test_versicolor);

    cout << "Sample {1.4, 0.2} Predicted: " << pred_setosa << " (Expected: 0)" << endl;
    cout << "Sample {4.2, 1.4} Predicted: " << pred_versicolor << " (Expected: 1)" << endl;
}