BIN:=earthquake
BIN_OMP:=$(basename $(wildcard omp-*.c))
BIN_SIMD:=$(basename $(wildcard simd-*.c))
BIN_CUDA:=$(basename $(wildcard cuda-*.cu))
BINS:=$(BIN) $(BIN_OMP) $(BIN_CUDA) $(BIN_SIMD)

CFLAGS+=-std=c99 -Wall -Wpedantic -O3 -D_XOPEN_SOURCE=600
NVCCFLAGS+=-Wno-deprecated-gpu-targets -O3
LDLIBS+=-fopenmp -lgomp
SIMDFLAGS:=-march=native -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed -mavx2 #-ffast-math -funsafe-math-optimizations

.PHONY: clean plot

all: cuda omp simd serial

serial:
	cc $(CFLAGS) earthquake.c -o $(BIN)

omp:
	cc $(CFLAGS) $(LDLIBS) omp-earthquake.c -o $(BIN_OMP)

cuda:
	nvcc $(NVCCFLAGS) cuda-earthquake.cu -o $(BIN_CUDA)

simd:
	cc $(CFLAGS) $(LDLIBS) $(SIMDFLAGS) simd-earthquake.c -o $(BIN_SIMD)

plot: serial
	./$(EXE_SERIAL) 100000 256 > out
	gnuplot plot.gp

clean:
	\rm -f $(BINS) *.o *~ *.png out*

