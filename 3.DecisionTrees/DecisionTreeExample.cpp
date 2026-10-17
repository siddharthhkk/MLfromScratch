#include <bits/stdc++.h>
using namespace std;
#include "DecisionTree.h"

// g++ DecisionTreeExample.cpp DecisionTree.cpp -o DecisionTreeExample.exe && ./DecisionTreeExample.exe
int main(){
    vector<vector<double>> X_iris = {{1.4, 0.2}, {1.3, 0.2}, {1.5, 0.2}, {1.4, 0.3}, {1.7, 0.4},
                                               {4.7, 1.4}, {4.5, 1.5}, {4.9, 1.5}, {4.0, 1.3}, {4.6, 1.3}};
                                               // i is the data sample, j is the feature 
    vector<int> y_iris = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    shared_ptr<Node> root = grow_tree(X_iris, y_iris, 0, 3);
    vector<double> test_sample = {4.2, 1.4}; 
    cout << "Predicted Class for sample {4.2, 1.4}: " << predict(root, test_sample) << endl;
}