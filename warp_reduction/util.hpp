#pragma once

static size_t read_arg(int argc, char** argv, size_t index, int default_value) {
    if(argc>index) {
        try {
            auto n = std::stoi(argv[index]);
            if(n<0) {
                return default_value;
            }
            return n;
        }
        catch (std::exception e) {
            std::cout << "error : invalid argument \'" << argv[index]
                << "\', expected a positive integer." << std::endl;
            exit(-1);
        }
    }

    return default_value;
}

__device__
void bitstring(char *str, unsigned v) {
    str[32]=0;
    unsigned mask = 1;
    for(int i=31; i>=0; --i, mask<<=1) {
       str[i] = (v&mask ? '1' : '0');
    }
}

// return the power of 2 that is _less than or equal_ to i
__device__
unsigned rounddown_power_of_2(unsigned i) {
    // handle power of 2 and zero
    if(__popc(i)<2) return i;

    return 1u<<(31u - __clz(i));
}

__device__ __inline__
double get_from_lane(double x, unsigned lane) {
        // split the double number into 2 32b registers.
        int lo, hi;

        asm volatile( "mov.b64 { %0, %1 }, %2;" : "=r"(lo), "=r"(hi) : "d"(x) );

        // shuffle the two 32b registers.
        lo = __shfl(lo, lane);
        hi = __shfl(hi, lane);

        // return the recombined 64 bit value
        return __hiloint2double( hi, lo );
}

__device__ __inline__
float get_from_lane(float value, unsigned lane) {
    return __shfl(value, lane);
}

// the atomic update overloads for double are in the root namespace
// so that they match the namespace of the CUDA builtin equivalents
// for 32 bit float and 32/64 bit int
__device__ __inline__
static double atomicAdd(double* address, double val)
{
    using I = unsigned long long int;
    I* address_as_ull = (I*)address;
    I old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
                address_as_ull,
                assumed,
                __double_as_longlong(val + __longlong_as_double(assumed))
        );
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ __inline__
static double atomicSub(double* address, double val)
{
    using I = unsigned long long int;
    I* address_as_ull = (I*)address;
    I old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(
                address_as_ull,
                assumed,
                __double_as_longlong(__longlong_as_double(assumed) - val)
        );
    } while (assumed != old);
    return __longlong_as_double(old);
}

