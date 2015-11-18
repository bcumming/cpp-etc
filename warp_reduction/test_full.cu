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
    std::vector<device_vector> out(3);
    for(auto &o: out) {
        o = device_vector(m);
    }

    in(memory::all) = value_type{1};

    auto threads_per_block=256;
    auto blocks=(n+threads_per_block-1)/threads_per_block;

    for(auto &o: out) o(memory::all) = value_type{0};
    auto b1 = stream_compute.insert_event();
    gpu::reduce_by_index
        <<<blocks, threads_per_block>>>
        (in.data(), out[0].data(), p.data(), n);
    auto e1 = stream_compute.insert_event();
    e1.wait();
    std::cout << "  1  " << e1.time_since(b1) << " seconds" << std::endl;
    test(solution, host_vector(out[0]));

    for(auto &o: out) o(memory::all) = value_type{0};
    auto b2 = stream_compute.insert_event();
    gpu::reduce_by_index<value_type>
        <<<blocks, threads_per_block>>>
        (in.data(), out[0].data(), out[1].data(), p.data(), n);
    auto e2 = stream_compute.insert_event();
    e2.wait();
    std::cout << "  2  " << e2.time_since(b2) << " seconds" << std::endl;
    test(solution, host_vector(out[0]));
    test(solution, host_vector(out[1]));

    for(auto &o: out) o(memory::all) = value_type{0};
    auto b3 = stream_compute.insert_event();
    gpu::reduce_by_index<value_type>
        <<<blocks, threads_per_block>>>
        (in.data(), out[0].data(), out[1].data(), out[2].data(), p.data(), n);
    auto e3 = stream_compute.insert_event();
    e3.wait();
    std::cout << "  3  " << e3.time_since(b3) << " seconds" << std::endl;
    test(solution, host_vector(out[0]));
    test(solution, host_vector(out[1]));
    test(solution, host_vector(out[2]));

    return 0;
}

