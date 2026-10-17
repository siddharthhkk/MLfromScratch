#include <bits/stdc++.h>
using namespace std;
#include "DecisionTree.h"

double entropy(vector<int> &y){
    double p1 = 0;
    for(int i = 0; i < y.size(); i++){
        if(y[i] == 1){
            p1++;
        }
    }
    p1 = p1 / y.size();
    if(p1 == 0 || p1 == 1) return 0;
    double entropy = (-p1 * log2(p1)) - ((1 - p1) * log2(1 - p1));
    return entropy;
}
double information_gain(vector<vector<double>> &X, vector<int> &y, int feature_index, double threshold){
    double total_samples = y.size();
    double node_entropy = entropy(y);
    vector<int> left_y, right_y;
    for(int i = 0; i < total_samples; i++){
        if(X[i][feature_index] <= threshold) left_y.push_back(y[i]);
        else right_y.push_back(y[i]);
    }
    if (left_y.empty() || right_y.empty()) return 0.0;
    double left_w = left_y.size() / total_samples;
    double right_w = right_y.size() / total_samples;
    double left_entropy = entropy(left_y);
    double right_entropy = entropy(right_y);
    double weighted_entropy = (left_w * left_entropy) + (right_w * right_entropy);
    return node_entropy - weighted_entropy;;
}

double find_best_split(vector<vector<double>> &X, vector<int> &y, int &best_feature_index, double &best_threshold){
    double max_gain = -1;
    best_feature_index = -1;
    best_threshold = -1;
    int num_samples = X.size();
    int num_features = X[0].size();
    for(int j = 0; j < num_features; j++){
        for(int i = 0; i < num_samples; i++){
            double threshold = X[i][j];
            double gain = information_gain(X, y, j, threshold);
            if(gain > max_gain){
                max_gain = gain;
                best_feature_index = j;
                best_threshold = threshold;
            }
        }
    }
    return max_gain;
}

shared_ptr<Node> grow_tree(vector<vector<double>> &X, vector<int> &y, int depth, int max_depth){
    shared_ptr<Node> node = make_shared<Node>();
    int count1 = count(y.begin(), y.end(), 1);
    int majority_class = (count1 >= (y.size() - count1)) ? 1 : 0;

    if (depth >= max_depth || entropy(y) == 0 || y.empty()) {
        node->is_leaf = true;
        node->prediction = majority_class;
        return node;
    }
    int best_feat;
    double best_thresh;
    double max_gain = find_best_split(X, y, best_feat, best_thresh);
    if (max_gain <= 1e-6) {
        node->is_leaf = true;
        node->prediction = majority_class;
        return node;
    }

    node->feature_index = best_feat;
    node->threshold = best_thresh;
    vector<vector<double>> left_X, right_X;
    vector<int> left_y, right_y;
    for (int i = 0; i < X.size(); i++) {
        if (X[i][best_feat] <= best_thresh) {
            left_X.push_back(X[i]);
            left_y.push_back(y[i]);
        } 
        else {
            right_X.push_back(X[i]);
            right_y.push_back(y[i]);
        }
    }
    
    // Recursively grow branches
    node->left = grow_tree(left_X, left_y, depth + 1, max_depth);
    node->right = grow_tree(right_X, right_y, depth + 1, max_depth);
    return node;
}
int predict(shared_ptr<Node> node, vector<double> &x){
    if(node->is_leaf) return node->prediction;
    if(x[node->feature_index] <= node->threshold)return predict(node->left, x);
    else return predict(node->right, x);
}