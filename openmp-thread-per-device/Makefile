# Collective Communications for CUDA GPUs

EXES := test_broadcast test_scatter

NVCC ?= nvcc
MPICC ?= mpiCC

OPTFLAGS ?= -O3
CFLAGS ?= -Wall -fopenmp
NVCCFLAGS ?=

.PHONY: all
all: $(EXES)

CFLAGS += $(OPTFLAGS)
NVCCFLAGS += $(OPTFLAGS)

space := $() $()
comma := ,
CS_CFLAGS = $(subst $(space),$(comma),$(CFLAGS))

MPICCFLAGS := $(shell $(MPICC) --showme:compile)
CS_MPICCFLAGS := $(subst $(space),$(comma),$(MPICCFLAGS))

NVCCFLAGS += -Xcompiler $(CS_CFLAGS),$(CS_MPICCFLAGS)
CUDALIBPATH := $(strip $(shell nvcc -v dne.cu 2>&1 | grep LIBRARIES | cut -d = -f 2))

LDFLAGS := -L./collectives $(CUDALIBPATH)
LDLIBS := -lcudaMPI -lcudart -lgomp

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $^ -o $@

%: %.o
	$(MPICC) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean
clean:
	rm -rf *~ *.o $(EXES)
