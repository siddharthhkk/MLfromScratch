#include <bits/stdc++.h>
using namespace std;

struct Movie {
    int id;
    string title;
    vector<double> features;
};
struct RecommendedMovie {
    string title;
    double similarityScore;
};

bool compareRecommendations(const RecommendedMovie& a, const RecommendedMovie& b) {
    return a.similarityScore > b.similarityScore;
}

double cosineSimilarity(const vector<double>& vec1, const vector<double>& vec2) {
    double dotProduct = 0.0;
    double norm1 = 0.0;
    double norm2 = 0.0;

    for (int i = 0; i < vec1.size(); ++i) {
        dotProduct += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    if (norm1 == 0.0 || norm2 == 0.0) return 0.0;
    return dotProduct / (sqrt(norm1) * sqrt(norm2));
}

int main() {
    vector<Movie> movies = {
        {0, "The Matrix",      {1.0, 1.0, 0.0}}, 
        {1, "John Wick",       {1.0, 0.0, 0.0}},
        {2, "Superbad",        {0.0, 0.0, 1.0}},
        {3, "Interstellar",    {0.0, 1.0, 0.0}},
        {4, "Free Guy",        {1.0, 1.0, 1.0}}
    };
    int numGenres = 3;
    vector<double> userRatings = {4.0, 5.0, -1.0, -1.0, -1.0};

    vector<double> userProfile(numGenres, 0.0);
    double weightSum = 0.0;

    for (size_t i = 0; i < movies.size(); ++i) {
        if (userRatings[i] != -1.0) {
            double ratingWeight = userRatings[i];
            for (size_t g = 0; g < numGenres; ++g) {
                userProfile[g] += movies[i].features[g] * ratingWeight;
            }
            weightSum += ratingWeight;
        }
    }
    if (weightSum > 0.0) {
        for (size_t g = 0; g < numGenres; ++g) {
            userProfile[g] /= weightSum;
        }
    }

    cout << fixed << setprecision(2);
    cout << "=== Calculated User Taste Profile ===" << endl;
    cout << "Action: " << userProfile[0] << " | Sci-Fi: " << userProfile[1] << " | Comedy: " << userProfile[2] << "\n\n";

    vector<RecommendedMovie> recommendations;

    for (size_t i = 0; i < movies.size(); ++i) {
        if (userRatings[i] == -1.0) { // Only recommend unrated movies
            double similarity = cosineSimilarity(userProfile, movies[i].features);
            recommendations.push_back({movies[i].title, similarity});
        }
    }
    sort(recommendations.begin(), recommendations.end(), compareRecommendations);

    cout << "=== Top Content-Based Recommendations ===" << endl;
    for (size_t i = 0; i < recommendations.size(); ++i) {
        cout << i + 1 << ". " << recommendations[i].title 
             << " (Match Score: " << recommendations[i].similarityScore * 100 << "%)" << endl;
    }
}