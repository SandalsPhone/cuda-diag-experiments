CXX=g++
NVCC=nvcc

a: main.o diagonalEditDistance.o
	$(NVCC) -o a main.obj diagonalEditDistance.obj

main.o: main.cu
	$(NVCC) -c main.cu

diagonalEditDistance.o: diagonalEditDistance.cu
	$(NVCC) -c diagonalEditDistance.cu