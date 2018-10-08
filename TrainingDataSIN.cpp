#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

using namespace std;

int main()
{
    //Random training sets for sin(x)
    freopen("trainingData.txt","w",stdout);
    cout << "topology: 1 1000 1" << endl;
    for (int i = 4000; i >= 0; --i){
        double n1 = 6*rand ()/ double(RAND_MAX);
        double t = sin(n1);
        cout << "in: " << n1 <<  endl;
        cout << "out: " << t << endl;
    }
}
