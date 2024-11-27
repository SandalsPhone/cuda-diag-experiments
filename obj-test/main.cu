#include <iostream>
#include "diagonalEditDistance.h"

using namespace std;

int main(int argc, char *argv[]){

    string one = "ABCDA";
    string two = "CBAD";

    const char *X = one.data();
    const char *Y = two.data();

    int result = diagonalEditDistance(X, Y);

    printf("Result: %d", result);

}