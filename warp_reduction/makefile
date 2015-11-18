FLAGS=-arch=sm_35 -std=c++11
FLAGS+=-O3

all : test_implementation.exe test_full.exe

test_implementation.exe : test_implementation.cu util.hpp warp_reduction.hpp
	nvcc ${FLAGS} test_implementation.cu vector/lib/libgpu.a -o test_implementation.exe

test_full.exe : test_full.cu util.hpp warp_reduction.hpp
	nvcc ${FLAGS} test_full.cu vector/lib/libgpu.a -o test_full.exe

clean :
	rm -f test_implementation.exe test_full.exe *.o a.out
