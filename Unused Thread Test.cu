#include <cuda_runtime.h>
#include <iostream>

//just a minor check to make sure
//the idea is after thread 'x' the threads won't do anything
__global__ void printId(){
    int id = threadIdx.x + blockIdx.x*blockDim.x;
    int x = 10;

    //alternative way
    //if(id>=x){
    //    return;
    //}
    //printf("%d \n", id);

    if(id<x){
        printf("%d \n", id);
    }
}

int main(int argc, char *argv[]){
    int threadSize = 256;

    printId<<<1,threadSize>>>();

    return 0;
}