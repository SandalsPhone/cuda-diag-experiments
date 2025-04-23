#include <iostream>
#include <chrono>
//#include "diagonalEditDistance.h"
#include <vector>

using namespace std;

__global__ void insertValues(const char *X, const char *Y, int *arr, int slice, int z, int rowLength, int colLength){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    //row gets shifted down depending on the value of z
    int row = id + z;

    int col;
    //col starts with slice until its higher than column length
    if(slice < colLength){
        col = slice - id;
    }
    else{
        col = (colLength - 1) - id;
    }
    if(col<0){
        return;
    }
    //the position in 1D array is calculated with (row x column length) + column
    int pos = (row * colLength) + col;

    if (col == 0) {
		arr[pos] = row;
	}
	else if (row == 0) {
		arr[pos] = col;
	}
	//else if (X[col-1] == Y[row-1]) {
    else if (X[row-1] == Y[col-1]) {
		arr[pos] = arr[pos - colLength - 1];
	}
	else {
	//dp[row - 1][col] og 2nd min
		arr[pos] = 1 + min(min(arr[pos - 1], arr[pos - colLength]), arr[pos - colLength - 1]);
	}  
    
}


int diagonalEditDistance(const char *X, const char *Y, char *deviceX, char *deviceY, int *dp, int *arr){
        //testing 2D arrays with 1D array representation too
    //int *arr;


    //char X[] = "ABCD";
    //char Y[] = "CBAD";
    //int rowLength = strlen(X);
    //int colLength = strlen(Y);


    cudaMemcpy(deviceX, X, strlen(X)*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceY, Y, strlen(Y)*sizeof(char), cudaMemcpyHostToDevice);

    int rowLength = strlen(X) + 1;
    int colLength = strlen(Y) + 1;
    

    int size = rowLength*colLength;

    //allocate memory
    //dp = (int*) std::malloc(size*sizeof(int));



    //just as a note:
    //the for loop below for the diagonal implementation uses slice as the baseline
    //slice functions as to determine the position of the diagonal for the iteration
    //example:
    //lets say slice = 1, visually on a 3x3 array its like this:
    //| 0  s  0 |
    //| s  0  0 |  <-- with s being the representation of the slice
    //| 0  0  0 |
    //
    //this continues until slice is higher the length of the column,
    //where the diagonal cannot be calculated with only the column as reference
    //the diagonal needs to continue through the 'bottom' half of the array 
    //
    //this is where z starts to function
    //after the slice is higher than the column, z is calculated using this formula:
    //z = slice - column + 1
    //lets say the slice is only 1 higher than the column,
    //this means the z is 1, and with that, the diagonal "shifts down" by 1
    //an example on a 3x3 array with z = 1:
    //| 0  0  0 |
    //| 0  0  s |
    //| 0  s  0 |

    int z, tSize;
    int bSize = 1;

	for(int slice=0; slice < (colLength+rowLength); slice++){
		if(slice < colLength){
			z = 0;
            tSize = colLength+rowLength;
		}
		else{
			z = slice - colLength + 1;
            tSize = rowLength - z;
		}
		
        //calculate thread and blocks used
		if(tSize <= 256){
			bSize = 1;
		}
		else{
			bSize = tSize / 256;
            tSize = 256;
		}

		insertValues<<<bSize, tSize>>>(deviceX, deviceY, arr, slice, z, rowLength, colLength);
	}

    //copy device array from insertValues to host array
    cudaMemcpy(dp, arr, size*sizeof(int), cudaMemcpyDeviceToHost);

    //printArr<<<1,1>>>(arr, rowLength, colLength);


    //int out;
    //cudaMemcpy(&out, arr+(rowLength*colLength) - 2, sizeof(int), cudaMemcpyDeviceToHost);

    //printArr<<<1,1>>>(arr, rowLength, colLength);

    //cudaFree(deviceX);
    //cudaFree(deviceY);
    //cudaFree(arr);
    //free(dp);
    
    //printf("%i",out);
    return dp[size-1];
}

int EditDistanceArray(const string& X, const string& Y, int m, int n, vector<vector<int> >& dp) {
	for (int i = 0; i <= m; i++) {
		for (int j = 0; j <= n; j++) {
			if (i == 0) {
				dp[i][j] = j;
			}
			else if (j == 0) {
				dp[i][j] = i;
			}
			else if (X[i - 1] == Y[j - 1]) {
				dp[i][j] = dp[i - 1][j - 1];
			}
			else {
				dp[i][j] = 1 + min(min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]);
			}
		}
	}
	return dp[m][n];
}

int main(int argc, char *argv[]){

    //string one = "GGGSSSSSSTTTTSFSFDAAAAAATGTTGCTTCCAAGTCTGAGGCGGCAAACAAGCCCATGAAGGCAAGCACTGACTAATTGTGCCAATGGGTGAACCGAGTTGGGGGACCAGTGTGTGCGGAAACCCGCATACGGCAGTACAAACGTTGTTAGACACATTCTGTAGCTTCGTTTCTCGGAGAACTGGTGTTTTGACCAGGGAGAAAGACCTCAATGCGTGTTT";
    string one = "GGGSSSSSSTTTTSFSFDAAAAAATGTTGCTTCCAAGTCTGAGGCGGCAAACAAGCCCATGAAGGCAAGCACTGACTAATTGTGCCAATGGGTGAACCG";
    //string two = "TTTTSSSSDSFSFSFSDFSDGGGGGCCCCCCCTTCTGCGGTCTGTGAATAGCGACAATATTGTCCCCTTTTGAAGTTCAATGATAGAAGTCCCTCAACAGTGGAGATAGCGAGCCTTGTGTGAAGATAGCTATAGGTCGAAAGTCCTTGCATTACTAGAGAATGATACGACGACGAGCCTATCCACAGACTGCCACCCTAAGTAAATGTATGTCACAAAGTGGCAACGTCGT";
    string two[2000];
    for(int i = 0; i<2000;i++){
        //two[i] = "TTTTSSSSDSFSFSFSDFSDGGGGGCCCCCCCTTCTGCGGTCTGTGAATAGCGACAATATTGTCCCCTTTTGAAGTTCAATGATAGAAGTCCCTCAACAGTGGAGATAGCGAGCCTTGTGTGAAGATAGCTATAGGTCGAAAGTCCTTGCATTACTAGAGAATGATACGACGACGAGCCTATCCACAGACTGCCACCCTAAGTAAATGTATGTCACAAAGTGGCAACGTCGT";
        two[i] = "TTTTSSSSDSFSFSFSDFSDGGGGGCCCCCCCTTCTGCGGTCTGTGAATAGCGACAATATTGTCCCCTTTTGAAGTTCAATGATAGAAGTCCCTCAACAG";
    }

    int result[2000];
    
    clock_t begin = clock();
    
    

    //const char *X = one.data();
    //const char *Y;

    //char *deviceX, *deviceY;
    //cudaMalloc(&deviceX, strlen(X)*sizeof(char));
    //cudaMalloc(&deviceY, (strlen(X)+10)*sizeof(char));

    //int *dp = (int*) std::malloc(((strlen(X)+1)*((strlen(X)+10))+1)*sizeof(int));
    //int *arr;
    //cudaMalloc(&arr, ((strlen(X)+1)*((strlen(X)+10))+1)*sizeof(int));

    int m = one.length();
    

    
    //int result = EditDistanceArray(one, two, m, n, dp);
    for(int i = 0;i<2000;i++){
        int n = two[i].length();
        vector<vector<int> > dp(m + 1, vector<int>(n + 1));
        //Y = two[i].data();
        //result[i] = diagonalEditDistance(X, Y, deviceX, deviceY, dp, arr);
        result[i] = EditDistanceArray(one, two[i], m, n, dp);
    }
    //result = diagonalEditDistance(X, Y);

    printf("Result 10: %d\n", result[1999]);

    //cudaFree(deviceX);
    //cudaFree(deviceY);
    //free(dp);
    //cudaFree(arr);

    clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout<< elapsed_secs<<endl;

}