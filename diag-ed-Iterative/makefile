CXX = g++
CFLAGS = -std=c++0x -O3 -g3 -Wall -c -fmessage-length=0 
RM = rm
NVCC=nvcc
NVCC_FLAGS=
# LINK_FLAGS= -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64" 

# ****************************************************


main: DNA.obj LCS2.obj EditDistance.obj Clone.obj Cluster2.obj LongestPath.obj CommonSubstring2.obj DividerBMA.obj DiagonalEditDistance.obj
	$(NVCC) -o main DNA.obj LCS2.obj EditDistance.obj Clone.obj Cluster2.obj LongestPath.obj CommonSubstring2.obj DiagonalEditDistance.obj
# ****************************************************

DNA.obj: DNA.cu
	$(NVCC) $(NVCC_FLAGS) -c DNA.cu

LCS2.obj: LCS2.cu
	$(NVCC) $(NVCC_FLAGS) -c LCS2.cu

EditDistance.obj: EditDistance.cu
	$(NVCC) $(NVCC_FLAGS) -c EditDistance.cu

Clone.obj: Clone.cu
	$(NVCC) $(NVCC_FLAGS) -c Clone.cu

Cluster2.obj: Cluster2.cu
	$(NVCC) $(NVCC_FLAGS) -c Cluster2.cu

LongestPath.obj: LongestPath.cu
	$(NVCC) $(NVCC_FLAGS) -c LongestPath.cu

CommonSubstring2.obj: CommonSubstring2.cu
	$(NVCC) $(NVCC_FLAGS) -c CommonSubstring2.cu

DividerBMA.obj: DividerBMA.cu
	$(NVCC) $(NVCC_FLAGS) -c DividerBMA.cu

DiagonalEditDistance.obj: DiagonalEditDistance.cu
	$(NVCC) $(NVCC_FLAGS) -c DiagonalEditDistance.cu


clean:
	$(RM) *.obj

