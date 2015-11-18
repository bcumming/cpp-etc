#define WITH_CUDA

#include <iostream>
#include <fstream>
#include <random>
#include <cstdio>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "vector/include/Vector.hpp"
#include "cudastreams/CudaEvent.h"
#include "cudastreams/CudaStream.h"
#include "util.hpp"
#include "warp_reduction.hpp"

using index_type = int;
using host_index   = memory::HostVector<index_type>;
using device_index = memory::DeviceVector<index_type>;

using value_type = double;
using host_vector   = memory::HostVector<value_type>;
using device_vector = memory::DeviceVector<value_type>;

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

template <typename T>
__global__
void reduce_by_index2(T* in, T* out, int* p, int n) {
    extern __shared__ T buffer[];

    auto tid = threadIdx.x + blockIdx.x*blockDim.x;
    auto lane_id = tid%32;

    auto right_limit = [] (unsigned roots, unsigned shift) {
        unsigned zeros_right  = __ffs(roots>>shift);
        return zeros_right ? shift -1 + zeros_right : 32;
    };

    if(tid<n) {
        // load index and value into registers
        int my_idx = p[tid];
        buffer[threadIdx.x] = in[tid];

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
            if(source_lane < rhs) {
                buffer[threadIdx.x] += buffer[threadIdx.x+width];
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
            auto sum = buffer[threadIdx.x];
            if(lane_id==0 || right==32) {
                atomicAdd(out+my_idx, sum);
            }
            else {
                out[my_idx] = sum;
            }
        }
    }
}

template <typename T>
__global__
void reduce_by_index3(T* in, T* out, int* p, int n) {
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

host_index generate_index(int n, int max_bucket_size) {
    std::random_device rd;
    std::default_random_engine e(rd());
    std::uniform_int_distribution<int> rng(1, max_bucket_size);

    std::cout << " == bucket size " << max_bucket_size << " ==" << std::endl;
    std::cout << " == array size " << n << " ==" << std::endl;

    // generate the index vector on the host
    host_index index(n);

    auto pos = 0;
    auto m = 0;
    while(pos<n) {
        auto increment = rng(e);
        auto final = std::min(pos+increment, n);
        while(pos<final) {
            index[pos++] = m;
        }
        ++m;
    }

    return index;
}

host_index read_index(std::string fname) {
    std::ifstream fid(fname);
    if(!fid.is_open()) {
        std::cerr << memory::util::red("error") << " : unable to open file "
                  << memory::util::yellow(fname) << std::endl;
        exit(1);
    }

    int n;
    fid >> n;
    std::cout << "loading index of length " << n << " from file " << fname << std::endl;
    host_index index(n);
    for(auto i=0; i<n; ++i) fid >> index[i];
    return index;
}

int main(int argc, char** argv) {
    int max_bucket_size = read_arg(argc, argv, 1, -1);

    // input  array of length n
    // output array of length m
    // sorted indexes in p (length n)
    auto ph =
        max_bucket_size < 1 ?
            read_index("index.txt")
          : generate_index(1<<25, max_bucket_size);
    const auto n = ph.size();
    auto m = ph[n-1];

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

    device_vector in(n);
    device_vector out(m);
    in(memory::all) = value_type{1};

    auto threads_per_block=256;
    auto blocks=(n+threads_per_block-1)/threads_per_block;

    // method 1 : naiive atomics
    //std::cout << "reduction using atomics... " << std::endl;
    out(memory::all) = value_type{0};
    auto b1 = stream_compute.insert_event();

    reduce_by_index1
        <<<blocks, threads_per_block, 0>>>
        (in.data(), out.data(), p.data(), n);

    auto e1 = stream_compute.insert_event();
    e1.wait();
    std::cout << "  naiive       " << e1.time_since(b1) << " seconds" << std::endl;

    test(solution, host_vector(out));

    // method 2 : reduction in shared memory
    //std::cout << "reduction using warp vote and reduction in shared memory... " << std::endl;
    out(memory::all) = value_type{0};

    auto shared_size = sizeof(value_type)*threads_per_block;
    auto b2 = stream_compute.insert_event();
    reduce_by_index2
        <<<blocks, threads_per_block, shared_size>>>
        (in.data(), out.data(), p.data(), n);
    auto e2 = stream_compute.insert_event();
    e2.wait();
    std::cout << "  in shared    " << e2.time_since(b2) << " seconds" << std::endl;
    test(solution, host_vector(out));

    // method 3 : reduction in registers
    //std::cout << "reduction using warp vote and reduction in registers... " << std::endl;
    out(memory::all) = value_type{0};

    auto b3 = stream_compute.insert_event();
    gpu::reduce_by_index
        <<<blocks, threads_per_block>>>
        (in.data(), out.data(), p.data(), n);
    auto e3 = stream_compute.insert_event();
    e3.wait();
    std::cout << "  in register  " << e3.time_since(b3) << " seconds" << std::endl;

    test(solution, host_vector(out));

    return 0;
}

