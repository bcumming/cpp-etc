#pragma once

#include "util.hpp"
#include "array.hpp"

namespace gpu {

    // Stores information about a run length.
    //
    // A run length is a set of identical adjacent indexes in an index array.
    //
    // When doing a parallel reduction by index each thread must know about
    // which of its neighbour threads are contributiing to the same global
    // location (i.e. which neighbours have the same index).
    //
    // The constructor takes the thread id and index of each thread
    // and the threads work cooperatively using warp shuffles to determine
    // their run length information, so that each thread will have unique
    // information that describes its run length and its position therein.
    struct run_length_type {
        unsigned left;
        unsigned right;
        unsigned width;
        unsigned lane_id;

        __device__ __inline__
        bool is_root() const {
            return left == lane_id;
        }

        __device__ __inline__
        bool may_cross_warp() const {
            return left==0 || right==32;
        }

        template <typename I1, typename I2>
        __device__
        run_length_type(I1 tid, I2 my_idx) {
            auto right_limit = [] (unsigned roots, unsigned shift) {
                unsigned zeros_right  = __ffs(roots>>shift);
                return zeros_right ? shift -1 + zeros_right : 32;
            };

            lane_id = tid%32;

            // determine if this thread is the root
            int left_idx  = __shfl_up(my_idx, 1);
            int is_root = 1;
            if(lane_id>0) {
                is_root = (left_idx != my_idx);
            }

            // determine the range this thread contributes to
            unsigned roots = __ballot(is_root);
            right = right_limit(roots, lane_id+1);
            left  = 31-right_limit(__brev(roots), 31-lane_id);
            width = rounddown_power_of_2(right - left);
        }
    };

    template <typename T, std::size_t N>
    __device__ __inline__
    void atomic_reduce_by_index(array<T,N> contributions, array<T*,N> targets, int my_idx, int tid) {
        run_length_type run(tid, my_idx);

        // get local copies of right and width, which are modified in the reduction loop
        auto rhs = run.right;
        auto width = run.width;

        while(width) {
            unsigned source_lane = run.lane_id + width;
            auto update_sum = [rhs, source_lane] (T &sum) {
                auto source_value = get_from_lane(sum, source_lane);
                if(source_lane < rhs) {
                    sum += source_value;
                }
            };

            for(auto &sum: contributions) {
                update_sum(sum);
            }

            rhs = run.left + width;
            width >>= 1;
        }

        if(run.is_root()) {
            // The first and last bucket in the warp have to be updated
            // automically in case they span multiple warps.
            // I experimented with further logic that only did an atomic update
            // on shared buckets, however the overheads of the tests were higher
            // than the atomics, even for double precision.
            if(run.may_cross_warp()) {
                #pragma unroll
                for(auto i=0; i<N; ++i) {
                    atomicAdd(targets[i]+my_idx, contributions[i]);
                }
            }
            else {
                #pragma unroll
                for(auto i=0; i<N; ++i) {
                    targets[i][my_idx] = contributions[i];
                }
            }
        }
    }

//#define READ_INPUT
    template <typename T>
    __global__
    void reduce_by_index(T* in, T* out, int* p, int n) {
        auto tid = threadIdx.x + blockIdx.x*blockDim.x;

        if(tid<n) {
            // load index and value into registers
            int my_idx = p[tid];
            #ifdef READ_INPUT
            atomic_reduce_by_index<T, 1>(in[tid], out, my_idx, tid);
            #else
            atomic_reduce_by_index<T, 1>(T{1}, out, my_idx, tid);
            #endif
        }
    }

    template <typename T>
    __global__
    void reduce_by_index(T* in, T* out1, T* out2, int* p, int n) {
        auto tid = threadIdx.x + blockIdx.x*blockDim.x;

        if(tid<n) {
            // load index and value into registers
            int my_idx = p[tid];
            array<T*, 2> outputs({out1, out2});
            #ifdef READ_INPUT
            array<T, 2> inputs(in[tid]);
            #else
            array<T, 2> inputs(T{1});
            #endif
            atomic_reduce_by_index<T, 2>(inputs, outputs, my_idx, tid);
        }
    }

    template <typename T>
    __global__
    void reduce_by_index(T* in, T* out1, T* out2, T* out3, int* p, int n) {
        auto tid = threadIdx.x + blockIdx.x*blockDim.x;

        if(tid<n) {
            // load index and value into registers
            int my_idx = p[tid];
            array<T*, 3> outputs({out1, out2, out3});
            #ifdef READ_INPUT
            array<T, 3> inputs(in[tid]);
            #else
            //array<T, 3> inputs(T{1});
            //array<T, 3> inputs({1.0,1.0,1.0});
            array<T, 3> inputs;
            inputs[0] = 1.;
            inputs[1] = 1.;
            inputs[2] = 1.;
            #endif
            atomic_reduce_by_index<T, 3>(inputs, outputs, my_idx, tid);
        }
    }

} // namespace gpu
