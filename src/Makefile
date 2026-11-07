NVCC ?= $(or $(shell command -v nvcc 2>/dev/null),$(HOME)/.cache/matrix-multiplication/cuda/bin/nvcc)
ARCH ?= 86
FLAGS = -O3 -std=c++17 -arch=sm_$(ARCH) -Xcompiler=-pthread,-Wall,-Wextra

.PHONY: all test clean
all: matmul

matmul: main.cpp Makefile
	"$(NVCC)" $(FLAGS) -x cu "$<" -o "$@"

test: matmul
	./matmul test

clean:
	rm -f matmul
