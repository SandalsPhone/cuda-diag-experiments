CXX = g++
CFLAGS = -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 
RM = rm
NVCC=nvcc
NVCC_FLAGS=
# LINK_FLAGS= -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64" 

# ****************************************************


main: DNA.o LCS2.o EditDistance.o Clone.o Cluster2.o LongestPath.o CommonSubstring2.o DividerBMA.o DiagonalEditDistance.o
	$(NVCC) -o main DNA.o LCS2.o EditDistance.o Clone.o Cluster2.o LongestPath.o CommonSubstring2.o DiagonalEditDistance.o
# ****************************************************

DNA.o: DNA.cu
	$(NVCC) $(NVCC_FLAGS) -c DNA.cu

LCS2.o: LCS2.cu
	$(NVCC) $(NVCC_FLAGS) -c LCS2.cu

EditDistance.o: EditDistance.cu
	$(NVCC) $(NVCC_FLAGS) -c EditDistance.cu

Clone.o: Clone.cu
	$(NVCC) $(NVCC_FLAGS) -c Clone.cu

Cluster2.o: Cluster2.cu
	$(NVCC) $(NVCC_FLAGS) -c Cluster2.cu

LongestPath.o: LongestPath.cu
	$(NVCC) $(NVCC_FLAGS) -c LongestPath.cu

CommonSubstring2.o: CommonSubstring2.cu
	$(NVCC) $(NVCC_FLAGS) -c CommonSubstring2.cu

DividerBMA.o: DividerBMA.cu
	$(NVCC) $(NVCC_FLAGS) -c DividerBMA.cu

DiagonalEditDistance.o: DiagonalEditDistance.cu
	$(NVCC) $(NVCC_FLAGS) -c DiagonalEditDistance.cu


clean:
	$(RM) *.o

