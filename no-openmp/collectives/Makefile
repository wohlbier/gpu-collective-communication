# Collective communication library for GPU

NVCC ?= nvcc
MPICC ?= mpiCC

OPTFLAGS := -O3
CFLAGS := -Wall $(OPTFLAGS)
NVCCFLAGS := $(OPTFLAGS)

SRCS :=	\
	init.cu \
	broadcast.cu \
	scatter.cu

LIBNAME := libcudaMPI.a

.PHONY: all
all: $(LIBNAME)

OBJS := $(SRCS:.cu=.o)

MPICCFLAGS := $(shell $(MPICC) --showme:compile)

# get cflags to be comma separated to pass through nvcc
space := $() $()
comma := ,
CS_CFLAGS = $(subst $(space),$(comma),$(CFLAGS))
CS_MPICCFLAGS := $(subst $(space),'$(comma)',$(MPICCFLAGS))

NVCCFLAGS += -Xcompiler $(CS_CFLAGS),$(CS_MPICCFLAGS)

$(LIBNAME): $(OBJS)
	$(NVCC) -lib $^ -o $@

%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $^ -o $@

clean:
	rm -rf *~ *.o $(LIBNAME)
