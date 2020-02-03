MPICC ?= mpiCC
NVCC ?= nvcc

CFLAGS ?= -Wall
NVCCFLAGS ?= -G
LDFLAGS ?=
LDLIBS ?= -lnccl -lcudart
OPTFLAGS ?= -O3 -g

EXES := nccl_broadcast nccl_allreduce nccl_scatter

##########

NVCCFLAGS += $(OPTFLAGS)
CFLAGS += $(OPTFLAGS)

space := $() $()
comma := ,
MPICCFLAGS := $(shell $(MPICC) --showme:compile)
CS_MPICCFLAGS := $(subst $(space),$(comma),$(MPICCFLAGS))
CS_CFLAGS := $(subst $(space),$(comma),$(CFLAGS))

NVCCFLAGS += -Xcompiler $(CS_MPICCFLAGS),$(CS_CFLAGS)

NVLIBRARIES := $(strip $(shell $(NVCC) -v dne.cu 2>&1 | grep LIBRARIES | cut -d = -f 2))
LDFLAGS += $(NVLIBRARIES)

.PHONY: all
all: $(EXES)

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $^ -o $@

%: %.o
	$(MPICC) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean
clean:
	$(RM) *~ *.o $(EXES)
