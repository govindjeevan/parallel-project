EXECUTABLE := run_nn

CU_FILES   := neuralNetwork.cu neuralNetworkTrainer.cu
# CU_FILES   := seqNeuralNetwork.cu

CU_DEPS    :=

CC_FILES   := main.cpp dataReader.cpp 
#neuralNetwork.cpp neuralNetworkTrainer.cpp

# LOGS	   := logs

###########################################################

ARCH=$(shell uname | sed -e 's/-.*//g')
OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-fopenmp -O3 -Wall -g
HOSTNAME=$(shell hostname)

LIBS       := 
FRAMEWORKS :=

# -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas
NVCCFLAGS=-O3 -m64 -Xcompiler -fopenmp --gpu-architecture compute_35
LIBS += GL glut cudart cublas
# openblas

ifneq ($(wildcard /opt/cuda-8.0/.*),)
# Latedays
LDFLAGS=-L/opt/cuda-8.0/lib64/ -lcudart
else
# GHC
LDFLAGS=-L/usr/local/cuda/lib64/ -lcudart
endif


LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/dataReader.o $(OBJDIR)/neuralNetworkTrainer.o $(OBJDIR)/neuralNetwork.o


.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS)

# check:	default
# 		./checker.pl

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
