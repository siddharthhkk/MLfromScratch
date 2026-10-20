#include <bits/stdc++.h>
using namespace std;

using RatingMatrix = vector<vector<double>>;

double cosineSimilarity(const vector<double>& u1, const vector<double>& u2) {
    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;
    for (int i = 0; i < u1.size(); ++i) {
        if (u1[i] != -1.0 && u2[i] != -1.0) {
            dotProduct += u1[i] * u2[i];
            norm1 += u1[i] * u1[i];
            norm2 += u2[i] * u2[i];
        }
    }
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    return dotProduct / (sqrt(norm1) * sqrt(norm2));
}
double getItemAverage(const RatingMatrix& matrix, int item) {
    double sum = 0.0;
    int count = 0;
    for (size_t u = 0; u < matrix.size(); ++u) {
        if (matrix[u][item] != -1.0) {
            sum += matrix[u][item];
            count++;
        }
    }
    return (count == 0) ? 0.0 : sum / count;
}
double predictRating(const RatingMatrix& matrix, int targetUser, int targetItem) {
    double weightedSum = 0.0;
    double similaritySum = 0.0;
    size_t numUsers = matrix.size();
    for (int u = 0; u < numUsers; ++u) {
        if (u == targetUser) continue;
        if (matrix[u][targetItem] != -1.0) {
            double sim = cosineSimilarity(matrix[targetUser], matrix[u]);
            if (sim > 0.0) {
                weightedSum += sim * matrix[u][targetItem];
                similaritySum += abs(sim);
            }
        }
    }
    if (similaritySum == 0.0) return getItemAverage(matrix, targetItem);
    return weightedSum / similaritySum;
}

int main(){
    RatingMatrix ratings = { // 4 users and 3 items
        { 5.0, 3.0, -1.0 }, 
        { 4.0, -1.0, 1.0 },
        { 1.0, 5.0,  4.0 },
        { -1.0, 4.0, 5.0 },
        {-1, -1, -1} // new user
    };

    int targetUser = 0;
    int targetItem = 2; // User 0 hasn't rated Item 2

    double predicted = predictRating(ratings, targetUser, targetItem);

    cout << fixed << setprecision(2);
    cout << "Predicted rating for new User " << targetUser 
         << " on Item " << targetItem << " is: " << predicted << endl;
}