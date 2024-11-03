//TODO: TEST OUT IF THE SLICE AND Z FOR THE DIAGONAL IMPLEMENTATION WORKS AS INTENDED
#include <cuda_runtime.h>
#include <string>
#include <iostream>
#include <string.h>

using namespace std;

__global__ void insertValues(char *X, char *Y, int *arr, int slice, int z, int rowLength, int colLength){
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

    //the position in 1D array is calculated with (row x column length) + column
    int pos = (row * colLength) + col;
    //arr[(row * colLength) + col] = slice;
    //printf("pos: (%d,%u) \n %i  \n", row, col, arr[(row*colLength) + col]);
    //printf("%i  \n", arr[(row*rowLength) + col]);


    if (col == 0) {
		arr[pos] = row;
	}
	else if (row == 0) {
		arr[pos] = col;
	}
	else if (X[col-1] == Y[row-1]) {
		arr[pos] = arr[pos - rowLength - 1];
        //arr[pos] = 2;
	}
	else {
	//dp[row - 1][col] og 2nd min
		arr[pos] = 1 + min(min(arr[pos - 1], arr[pos - rowLength]), arr[pos - rowLength - 1]);
        //arr[pos] = 1;
	}  
    
}

__global__ void printArr(int *arr, int rowLength, int colLength){
    int id = threadIdx.x + blockIdx.x*blockDim.x;

    printf("Array from kernel:\n");
    for(int i= 0; i<rowLength; i++){
        for(int j= 0; j<colLength; j++){
            int pos = (i*colLength)+j;
            printf("%d:%i  ", pos, arr[(i*colLength) + j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main(int argc, char *argv[]){
    //testing 2D arrays with 1D array representation too
    int *arr, *hostArr;


    char X[] = "ABCD";
    char Y[] = "CBAD";
    int rowLength = strlen(X);
    int colLength = strlen(Y);

    printf("X: %s\n", X);
    printf("Y: %s\n", Y);
    printf("X length: %i\n", rowLength);
    printf("Y length: %i\n", colLength);

    char *deviceX, *deviceY;
    cudaMalloc(&deviceX, rowLength*sizeof(char));
    cudaMalloc(&deviceY, colLength*sizeof(char));

    cudaMemcpy(deviceX, X, rowLength*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceY, Y, colLength*sizeof(char), cudaMemcpyHostToDevice);

    rowLength++;
    colLength++;

    

    int size = rowLength*colLength;

    //allocate memory
    hostArr = (int*) std::malloc(size*sizeof(int));
    cudaMalloc(&arr, size*sizeof(int));


    //print before running the kernel(s)
    printf("Array before:\n");
    for(int i= 0; i<rowLength; i++){
        for(int j= 0; j<colLength; j++){
            printf("%i  ", hostArr[(i*colLength) + j]);
        }
        printf("\n");
    }
    printf("\n");

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

    int z, tSize, lowestLength;
    int bSize = 1;
    if(colLength<rowLength){
        lowestLength = colLength;
    }
    else{
        lowestLength = rowLength;
    }

	for(int slice=0; slice < colLength*2; slice++){
		if(slice < colLength){
			z = 0;
            if(slice<lowestLength){
                tSize = slice + 1;
            }
            else{
                tSize = lowestLength;
            }
		}
		else{
			z = slice - colLength + 1;
            tSize = colLength - z;
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
    printf("\n");

    //copy device array from insertValues to host array
    cudaMemcpy(hostArr, arr, size*sizeof(int), cudaMemcpyDeviceToHost);

    //printArr<<<1,1>>>(arr, rowLength, colLength);
    
    //print after running the kernel(s)
    printf("Array after:\n");
    for(int i= 0; i<rowLength; i++){
        for(int j= 0; j<colLength; j++){
            //printf("%d,%u :%i  ", i, j, hostArr[(i*colLength) + j]);
            printf("%i  ", hostArr[(i*colLength) + j]);
        }
        printf("\n");
    }
    printf("\n");

    //free memory
    cudaFree(arr);
    free(hostArr);

    return 0;
}
