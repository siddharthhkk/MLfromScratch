#pragma once
struct Node{
    bool is_leaf = false;
    int prediction = -1;
    int feature_index = -1;
    double threshold = -1;
    shared_ptr<Node> left = nullptr;
    shared_ptr<Node> right = nullptr;
};

double information_gain(vector<vector<double>> &X, vector<int> &y, int feature_index, double threshold);
double entropy(vector<int> &y);
double find_best_split(vector<vector<double>> &X, vector<int> &y, int &best_feature_index, double &best_threshold);
shared_ptr<Node> grow_tree(vector<vector<double>> &X, vector<int> &y, int depth, int max_depth);
int predict(shared_ptr<Node> node, vector<double> &x);