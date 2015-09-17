#include <iostream>
#include <vector>

constexpr int x(int i) {
    return i ? i + x(i-1) : 0;
}

// this will produce fully unrolled code with constants inline
// in the assembly up to N=16 with gcc 5.2
template<int N>
int dot(int *y) {
    int sum = 0;
    #pragma unroll
    for(int i=0; i<N; ++i) {
        sum += y[i]*x(i);
    }
    return sum;
}

// this will produce fully unrolled code with constants inline
// in the assembly for higher N too (tested up to 24 with gcc 5.2)
template<int N>
int dot_T(int *y) {
    return x(N)*y[0] + dot_T<N-1>(y+1);
}

template<>
int dot_T<0>(int *y) {
    return x(0)*y[0];
}

void print(int N) {
    for(int i=0; i<N; ++i) {
        printf("%X\n", x(i));
    }
}

int main(void) {
    constexpr int N = 10;
    int y[N];

    //int result = dot<N>(&y[0]);
    int result = dot_T<N>(&y[0]);

    print(N);

    printf("%X\n", result);
    return result;
}
