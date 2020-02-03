NVCC=nvcc

NVCCFLAGS=-O2 -Xcompiler -Wall -std=c++11

EXES=nstream send ngpu_bandwidth simple_dual_copy nsend coop_multi_device scatter

.PHONY: all
all: $(EXES)

%.o: %.cu common.h
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

%: %.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)

.PHONY: clean
clean:
	$(RM) *~ *.o $(EXES)
