# This is a make file for building CNTweb program

NVCOMP = nvcc
NVFLAGS = -DSearchUM -maxrregcount=32 
#--relocatable-device-code=true

CXX    = gcc
CUDA_INCLUDE = /usr/local/cuda/include/
ADD_LIBS = -ltiff 

#CFLAGS = --std=c++11 -O3 -use_fast_math
CFLAGS = --std=c++11 -O3 -DSearchUM -use_fast_math

DEBFLAGS = -O0 -g -D_DEBUG
COMPFLAGS =


#OBJS   = NanoFiller.o IO.o kdTree.o simPrms_device.o simPrms.o select_device.o grscl_TIFFwrite.o plot_kernel.o statistics_collection.o cudaMain.o stateSR.o get_contact_arrays.o get_contact_data.o check_screened_Interactions.o core_cell_interaction_kernel.o adjacency_graph.o conductivity.o
INCDIR = ./inc -I$(CUDA_INCLUDE)
SRCDIR = ./src
SRCS   = $(notdir $(wildcard ${SRCDIR}/*.*))
OBJS	= $(addsuffix .o,$(basename ${SRCS}) )

all:	COMPFLAGS = $(CFLAGS)
all:	NanoFiller

NanoFiller:	$(OBJS)
	$(NVCOMP) $(NVFLAGS) $(COMPFLAGS) $(EXTFLAGS) -I$(INCDIR) $(ADD_LIBS) $^ -o $@


%.o:	$(SRCDIR)/%.cpp
	$(CXX) -c $(COMPFLAGS) -I$(INCDIR) $(ADD_LIBS) $< -o $@

%.o:	$(SRCDIR)/%.cu
	$(NVCOMP) -c $(NVFLAGS) -I$(INCDIR) $(ADD_LIBS) $< -o $@

debug:	COMPFLAGS  = -DDEBUG $(DEBFLAGS)
debug:	NanoFiller

clean:
	rm *.o
	rm NanoFiller
