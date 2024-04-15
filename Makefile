# *********************************************
# *** UNCOMMENT ONE OF THE FOLLOWING BLOCKS *** 
# *********************************************

# ***************************************************
# *** LUMI ***
CXX = hipcc 
CXXFLAGS = --offload-arch=gfx90a --std=c++17 -O3 -I"${MPICH_DIR}/include"  -I"/opt/rocm-5.2.3/include/" #-I"/opt/rocm/include" -I"/opt/rocm/rocprim/include/rocprim" -I"/opt/rocm/hipcub/include/hipcub/"
CXXFLAGS += -w -fopenmp
CXXFLAGS += -DUSE_CUDA
LDFLAGS = -L"${MPICH_DIR}/lib" -lmpi -L"/opt/rocm-5.2.3/lib/" -lhipblas -lhipsparse -lhipsolver  -L"/opt/rocm-5.2.3/rocprim/lib" -lrocm_smi64 -lrocsparse -lrocsolver -lrocblas

# ***************************************************

SRCDIR = src
SRCDIR_CG = dist_iterative
OBJDIR = obj
BINDIR = bin

TARGET = $(BINDIR)/runKMC																
all: $(TARGET) 

CUFILES = $(wildcard $(SRCDIR)/*.cu)
CPPFILES = $(wildcard $(SRCDIR)/*.cpp)		
CUFILES_CG = $(wildcard $(SRCDIR_CG)/*.cu)
CPPFILES_CG = $(wildcard $(SRCDIR_CG)/*.cpp)

CU_OBJ_FILES = $(patsubst $(SRCDIR)/%.cu, $(OBJDIR)/%.o, $(CUFILES))
CPP_OBJ_FILES = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPPFILES))
CU_OBJ_FILES_CG = $(patsubst $(SRCDIR_CG)/%.cu, $(OBJDIR)/%.o, $(CUFILES_CG))
CPP_OBJ_FILES_CG = $(patsubst $(SRCDIR_CG)/%.cpp, $(OBJDIR)/%.o, $(CPPFILES_CG))

DEPS = $(SRCDIR)/random_num.h $(SRCDIR)/input_parser.h

$(info MAKE INFO: Compile target - KMC Simulation ./bin/runKMC)
$(TARGET): $(CU_OBJ_FILES) $(CPP_OBJ_FILES) $(CU_OBJ_FILES_CG) $(CPP_OBJ_FILES_CG)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(DEPS) $(SRCDIR)/gpu_solvers.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR_CG)/%.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR)/%.cu $(DEPS) $(SRCDIR)/gpu_solvers.h
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(OBJDIR)/%.o: $(SRCDIR_CG)/%.cu
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean tests
