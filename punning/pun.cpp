#include <cstdint>

#include <algorithm>
#include <iostream>
#include <type_traits>

// we have to do some picky type punning to use the fill functions for all
// POD types with {8, 16, 32, 64} bits
#define FILL(N) \
void fill ## N(uint ## N ##_t* ptr, uint ## N ##_t val, size_t n) { \
    for(auto i=0; i<n; ++i) { \
        ptr[i] = val; \
    } \
} \
template <typename T> \
typename std::enable_if<sizeof(T)==sizeof(uint ## N ## _t)>::type \
fill(T* ptr, T value, size_t n) { \
    using I = uint ## N ## _t; \
    I v; \
    if(alignof(T)==alignof(I)) { \
        *reinterpret_cast<T*>(&v) = value; \
    } \
    else { \
        std::copy_n( \
            reinterpret_cast<char*>(&value), \
            sizeof(T), \
            reinterpret_cast<char*>(&v) \
        ); \
    } \
    fill ## N(reinterpret_cast<I*>(ptr), v, n); \
}

FILL(8)
FILL(16)
FILL(32)
FILL(64)

void test_float() {
    float farray[] = {0, 0, 0, 0, 0, 0};
    fill(farray, -1.f, 6);
    for(auto i: farray) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void test_double() {
    double darray[] = {0, 0, 0, 0, 0, 0};
    fill(darray, -2.,  6);
    for(auto i: darray) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void test_int() {
    int iarray[] = {0, 0, 0, 0, 0, 0};
    fill(iarray, -3,  6);
    for(auto i: iarray) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

void test_char() {
    char carray[] = {0, 0, 0, 0, 0, 0};
    fill<char>(carray, 'z',  6);
    for(auto i: carray) {
        std::cout << i << " ";
    }
    std::cout << std::endl;
}

int main(void) {
    test_float();
    test_double();
    test_int();
    test_char();
}

