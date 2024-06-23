NVCC = /usr/local/cuda/bin/nvcc
CC = g++

CUDA_HOME = /usr/local/cuda

CCFLAGS = -O3 -g -Wall -Wextra -fopenmp -std=c++17 -I"${CUDA_HOME}/include" 
CCFLAGS += -I"/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi" -I"/usr/lib/x86_64-linux-gnu/openmpi/include"

NVCCFLAGS = -O3 -arch=sm_60 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -std=c++17 -Xcompiler -fopenmp -I"${CUDA_HOME}/include" 
LDFLAGS = -L"${MPICH_DIR}/lib" -L"${CUDA_HOME}/lib64" 
LDFLAGS += -L"/usr/lib/x86_64-linux-gnu/openmpi/lib"
LDLIBS = -Wl,--copy-dt-needed-entries -lcuda -lcudart -lcublas -lcusolver -lcusparse -lm -lmpi -lmpi_cxx

SOURCES = main.cpp
SOURCES += utils.cpp utils_gpu.cu 
OWN_DIR = ../dist_iterative
SOURCES += $(wildcard $(OWN_DIR)/*.cpp) $(wildcard $(OWN_DIR)/*.cu)

CPP_SOURCES = $(filter %.cpp, $(SOURCES))
CU_SOURCES = $(filter %.cu, $(SOURCES))

CPP_OBJ_FILES=$(CPP_SOURCES:.cpp=.o)
CU_OBJ_FILES =$(CU_SOURCES:.cu=.o)
BINARY = main

.PHONY: all
all: $(BINARY)

$(BINARY): $(CPP_OBJ_FILES) $(CU_OBJ_FILES)
	$(CC) $(CCFLAGS) $(CPP_OBJ_FILES) $(CU_OBJ_FILES) -o $@ $(LDFLAGS) $(LDLIBS)

# Rule for compiling C++ source files
%.o: %.cpp
	$(CC) $(CCFLAGS) -c $< -o $@

# Rule for compiling CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(BINARY) *.o
	rm $(OWN_DIR)/*.o
