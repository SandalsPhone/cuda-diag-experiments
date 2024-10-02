#include <cuda_runtime.h>
#include <iostream>

__global__ void insertValues(int *arr){
    //Calculate id with:
    //threadIdx = id of thread
    //blockIdx = id of block
    //blockDim = number of threads in a block
    //assuming it's 1D
    int id = threadIdx.x + blockIdx.x*blockDim.x;

    arr[id] = id;
}

__global__ void printArr(int *arr){
    int id = threadIdx.x + blockIdx.x*blockDim.x;

    printf("%d  ", arr[id]);
}

int main(int argc, char *argv[]){
    //int *arr = new int[4];
    //have not tried 'new' yet
    int *arr, *hostArr;

    int size = 6;

    //allocate memory
    hostArr = (int*) std::malloc(size*sizeof(int));
    cudaMalloc(&arr, size*sizeof(int));

    //print before running the kernel
    printf("Array before:\n");
    for(int i= 0; i<size; i++){
        printf("%i  ", hostArr[i]);
    }
    printf("\n");


    insertValues<<<1,size>>>(arr);
    //printArr<<<1,size>>>(arr);
    cudaMemcpy(hostArr, arr, size*sizeof(int), cudaMemcpyDeviceToHost);

    //print after running the kernel
    printf("Array after:\n");
    for(int i= 0; i<size; i++){
        printf("%i  ", hostArr[i]);
    }


    //free memory
    cudaFree(arr);
    free(hostArr);

    //for when i try 'new' someday
    //delete hostArr;

    return 0;
}