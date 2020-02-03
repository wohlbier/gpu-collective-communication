# makefile to compile original code

EXES := bucket_all2all bucket_a2a_kernels

NVCC := nvcc

OPTFLAGS := -O2
CFLAGS := -Wall $(OPTFLAGS)
NVCCFLAGS := $(OPTFLAGS)

space := $() $()
comma := ,
CS_CFLAGS := $(subst $(space),$(comma),$(CFLAGS))

NVCCFLAGS += -Xcompiler $(CS_CFLAGS)

.PHONY: all
all: $(EXES)

bucket_a2a_kernels: bucket_a2a_kernels.o

bucket_all2all: bucket_all2all.o

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $^ -o $@

%: %.o
	$(NVCC) $(LDFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean
clean:
	rm -rf *~ *.o $(EXES)
