#include <iostream>
#include <vector>
#include <string>

using namespace std;

int n;
string seq;
vector<vector<int>> dp;
vector<int> pairs;


// Function to calculte the Nussinov score for a given sequence

bool complementary(char X, char Y)
{
    return (X == 'A' && Y == 'U') || (X == 'U' && Y == 'A') || (X == 'C' && Y == 'G') || (X == 'G' && Y == 'C') || (X == 'G' && Y == 'U') || (X == 'U' && Y == 'G');
}

int nussinov(){
    dp.resize(n, vector<int>(n, 0));
    
    //Dynamic programming loop 
    for (int l=4; l<n; l++){
        for (int i=0; i<n-4; i++){
            int j = i+l;

            if (j<n){
                //Insert gap
                int score = max(dp[i+1][j], dp[i][j-1]);

                //Base pair 
                if (complementary(seq[i], seq[j])) score = max(score, dp[i+1][j-1]+1);

                //Bifurcation
                for (int k=i+1; k<j; k++){
                    score = max(score, dp[i][k] + dp[k+1][k]);
                }

                dp[i][j] = score;
            }
            
        }
    }

    return dp[0][n-1];
}

string backtrack(int i, int j){
    if (j-i < 4){
        return string(j-i+1, '.');
    }

    if (dp[i][j] == dp[i+1][j]){
        return "." + backtrack(i+1, j);

    }

    if (dp[i][j] == dp[i][j-1]){
        return backtrack(i, j-1) + ".";
    }    

    if (complementary(seq[i], seq[j]) && dp[i][j] == dp[i+1][j-1] + 1){
        pairs[i] = j;
        pairs[j] = i;
        return '('+ backtrack(i+1, j-1) +')';
    }

    for (int k=i+1; k<j; k++){
        if (dp[i][j] == dp[i][k] + dp[k+1][j]){
            return backtrack(i, k) + backtrack(k+1, j);
        }
    }

    return "FAIL"; // should never happen!
}



int main(int argc, char* argv[])
{
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <sequence>" << endl;
        return 1;
    }

    seq = argv[1];
    n = seq.length();

    for (int i=0; i<n; ++i){
        pairs.push_back(i);
    }

    nussinov();
    
    backtrack(0, n-1);

    for (int i = 0; i< pairs.size(); ++i){
        cout << pairs[i] << " ";
    }
    cout << endl;

    return 0;
}
