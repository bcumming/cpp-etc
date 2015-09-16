FLAGS=-arch=sm_35 -std=c++11
FLAGS+=-O3
test : test.cu util.hpp
	nvcc ${FLAGS} test.cu vector/lib/libgpu.a -o test

clean :
	rm -f test *.o a.out
