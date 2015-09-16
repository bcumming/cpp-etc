#define WITH_CUDA

#include <iostream>
#include <random>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "vector/include/Vector.hpp"
#include "cudastreams/CudaEvent.h"
#include "cudastreams/CudaStream.h"
#include "util.hpp"

using index_type = int;
using host_index   = memory::HostVector<index_type>;
using device_index = memory::DeviceVector<index_type>;

using value_type = double;
using host_vector   = memory::HostVector<value_type>;
using device_vector = memory::DeviceVector<value_type>;

__device__
void bitstring(char *str, unsigned v) {
    str[32]=0;
    unsigned mask = 1;
    for(int i=31; i>=0; --i, mask<<=1) {
       str[i] = (v&mask ? '1' : '0');
    }
}

template <typename T>
__global__
void reduce_by_index1(T* in, T* out, int* p, int n) {
    auto tid = threadIdx.x + blockIdx.x*blockDim.x;

    if(tid<n) {
        // load index into register
        auto my_idx = p[tid];
        atomicAdd(out+my_idx, in[tid]);
    }
}

// return the power of 2 that is _less than or equal_ to i
__device__
unsigned rounddown_power_of_2(unsigned i) {
    // handle power of 2 and zero
    if(__popc(i)<2) return i;

    return 1u<<(31u - __clz(i));
}

template <typename T>
__global__
void reduce_by_index2(T* in, T* out, int* p, int n) {
    auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    auto lane_id = tid%32;

    auto right_limit = [] (unsigned roots, unsigned shift) {
        unsigned zeros_right  = __ffs(roots>>shift);
        return zeros_right ? shift -1 + zeros_right : 32;
    };

    if(tid<n) {
        // load index and value into registers
        int my_idx = p[tid];
        T sum = in[tid];

        // am I the root for an index?
        int left_idx  = __shfl_up(my_idx, 1);
        int is_root = 1;
        if(lane_id>0) {
            is_root = (left_idx != my_idx);
        }

        // determine the range I am contributing to
        unsigned roots = __ballot(is_root);
        unsigned right = right_limit(roots, lane_id+1);
        unsigned left  = 31-right_limit(__brev(roots), 31-lane_id);
        unsigned run_length = right - left;
        // keep a copy of rhs because right is modified in the reduction loop
        unsigned rhs = right;

        auto width = rounddown_power_of_2(run_length);
        while(width) {
            unsigned source_lane = lane_id + width;
            auto source_value = get_from_lane(sum, source_lane);
            if(source_lane < rhs) {
                sum += source_value;
            }
            rhs = left + width;
            width >>= 1;
        }

        if(is_root) {
            // The first and last bucket in the warp have to be updated
            // automically in case they span multiple warps.
            // I experimented with further logic that only did an atomic update
            // on shared buckets, however the overheads of the tests were higher
            // than the atomics, even for double precision.
            if(lane_id==0 || right==32) {
                atomicAdd(out+my_idx, sum);
            }
            else {
                out[my_idx] = sum;
            }
        }
    }
}

bool test(host_vector const& reference, host_vector const& v) {
    assert(reference.size() == v.size());
    auto success = true;
    for(auto i=0; i<reference.size(); ++i) {
        if(reference[i] != v[i]) {
            printf("  error %10d expected %5.1f got %5.1f\n",
                   (int)i, (float)(reference[i]), (float)(v[i]));
            success = false;
        }
    }

    return success;
}

void print(host_vector const& v) {
    auto pos = 0;
    while(pos<v.size()) {
        auto col = 0;
        while(col<32 && pos<v.size()) {
            printf("%3.0f", v[pos]);
            ++pos;
            ++col;
        }
        printf("\n");
    }
}

void print(host_index const& v) {
    auto pos = 0;
    while(pos<v.size()) {
        auto col = 0;
        while(col<32 && pos<v.size()) {
            printf("%3d", v[pos]);
            ++pos;
            ++col;
        }
        printf("\n");
    }
}

int main() {
    // input  array of length n
    // output array of length m
    // sorted indexes in p (length n)
    constexpr auto n = 1<<25;

    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_int_distribution<int> rng(1, 100);

    // generate the index vector on the host
    host_index ph(n);

    auto pos = 0;
    auto m = 0;
    while(pos<n) {
        auto increment = rng(e);
        auto final = std::min(pos+increment, n);
        while(pos<final) {
            ph[pos++] = m;
        }
        ++m;
    }

    // make reference solution
    host_vector solution(m);
    solution(0,m) = 0;
    for(auto i : ph) {
        solution[i] += 1;
    }

    if(n<=256) {
        std::cout << "in \n"; print(ph);
        std::cout << "out\n"; print(solution);
    }

    // configure cuda stream for timing
    CudaStream stream_compute(false);

    // push index to the device
    device_index p = ph;

    std::cout << "generated index and reference solution" << std::endl;

    device_vector in(n);
    device_vector out(m);
    in(memory::all) = value_type{1};

    auto threads_per_block=192;
    auto blocks=(n+threads_per_block-1)/threads_per_block;

    std::cout << "reduction using atomics... " << std::endl;
    out(memory::all) = value_type{0};
    auto b1 = stream_compute.insert_event();
    reduce_by_index1
        <<<blocks, threads_per_block, 0>>>
        (in.data(), out.data(), p.data(), n);
    auto e1 = stream_compute.insert_event();
    e1.wait();
    std::cout << "  " << e1.time_since(b1) << " seconds" << std::endl;

    test(solution, host_vector(out));

    std::cout << "reduction using warp vote and reduction in shared memory... " << std::endl;
    out(memory::all) = value_type{0};

    auto b2 = stream_compute.insert_event();
    reduce_by_index2
        <<<blocks, threads_per_block>>>
        (in.data(), out.data(), p.data(), n);
    auto e2 = stream_compute.insert_event();
    e2.wait();
    std::cout << "  " << e2.time_since(b2) << " seconds" << std::endl;

    //test(solution, host_vector(out));

    return 0;
}

