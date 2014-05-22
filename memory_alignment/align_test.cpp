#include <stdlib.h>

#include <iostream>
#include <type_traits>
#include <string>
#include <sstream>

template <typename T>
struct threepack {
    T v1;
    T v2;
    T v3;
};

template <typename T>
struct fourpack {
    T v1;
    T v2;
    T v3;
    T v4;
};

template <typename T>
struct type_printer {
    static std::string print () {
        std::stringstream s;
        s << "unknown(" << sizeof(T) << ")";
        return s.str();
    }
};

template <>
struct type_printer<char> {
    static std::string print () {
        return std::string("char");
    }
};

template <>
struct type_printer<int> {
    static std::string print () {
        return std::string("int");
    }
};

template <>
struct type_printer<float> {
    static std::string print () {
        return std::string("float");
    }
};

template <>
struct type_printer<double> {
    static std::string print () {
        return std::string("double");
    }
};

template <typename T>
struct type_printer<threepack<T>> {
    static std::string print () {
        std::stringstream s;
        s << "threepack<" << type_printer<T>::print() << ">";
        return s.str();
    }
};

template <typename T>
struct type_printer<fourpack<T>> {
    static std::string print () {
        std::stringstream s;
        s << "fourpack<" << type_printer<T>::print() << ">";
        return s.str();
    }
};

// meta function that returns true if x is a power of two (including 1)
template <size_t x>
struct is_power_of_two : std::integral_constant< bool, !(x&(x-1)) > {};

// meta function that returns the smallest power of two that is strictly greater than x
template <size_t x, size_t p=1>
struct next_power_of_two : std::integral_constant< size_t, next_power_of_two<x-(x&p), (p<<1) >::value> {};
template <size_t p>
struct next_power_of_two<0,p> : std::integral_constant<size_t, p> {};

// metafunction that returns the smallest power of two that is greater than or equal to x
template <size_t x>
struct round_up_power_of_two
    : std::integral_constant< size_t, is_power_of_two<x>::value ? x : next_power_of_two<x>::value >
{};

// metafunction that returns the smallest power of two that is greater than or equal to x,
// and greater than or equal to sizeof(void*)
template <size_t x>
struct minimum_possible_alignment
{
    static const size_t pot = round_up_power_of_two<x>::value;
    static const size_t value = pot < sizeof(void*) ? sizeof(void*) : pot;
};

// function that allocates memory with alignment specified as a template parameter
template <typename T, size_t alignment=minimum_possible_alignment<sizeof(T)>::value >
T* aligned_malloc(size_t size) {
    // double check that alignment is a multiple of sizeof(void*), as this is a prerequisite
    // for posix_memalign()
    static_assert( !(alignment%sizeof(void*)),
            "alignment is not a multiple of sizeof(void*)");
    static_assert( is_power_of_two<alignment>::value,
            "alignment is not a power of two");
    void *ptr;
    int result = posix_memalign(&ptr, alignment, size*sizeof(T));
    if(result)
        ptr=nullptr;
    return reinterpret_cast<T*>(ptr);
}

template <typename T>
bool test_align(T *ptr, size_t  align=minimum_possible_alignment<sizeof(T)>::value) {
    if(ptr==nullptr)
        return false;

    size_t t = reinterpret_cast<size_t>(ptr);

    return !(t&(align-1));
}

template <typename T>
bool align_by_type(size_t size) {
    T *ptr = aligned_malloc<T>(size);

    bool success = test_align(ptr);
    std::cout << ((success&&ptr!=nullptr) ? "good" : "bad ") << " for " << (type_printer<T>::print()) << std::endl;;
    return success;
}


int main(void) {
    static_assert(minimum_possible_alignment<1>::value == 8, "bad alignment calculated");
    static_assert(minimum_possible_alignment<2>::value == 8, "bad alignment calculated");
    static_assert(minimum_possible_alignment<3>::value == 8, "bad alignment calculated");
    static_assert(minimum_possible_alignment<4>::value == 8, "bad alignment calculated");
    static_assert(minimum_possible_alignment<5>::value == 8, "bad alignment calculated");
    static_assert(minimum_possible_alignment<6>::value == 8, "bad alignment calculated");
    static_assert(minimum_possible_alignment<7>::value == 8, "bad alignment calculated");
    static_assert(minimum_possible_alignment<8>::value == 8, "bad alignment calculated");
    static_assert(minimum_possible_alignment<9>::value == 16, "bad alignment calculated");
    static_assert(minimum_possible_alignment<10>::value == 16, "bad alignment calculated");
    static_assert(minimum_possible_alignment<11>::value == 16, "bad alignment calculated");
    static_assert(minimum_possible_alignment<12>::value == 16, "bad alignment calculated");
    static_assert(minimum_possible_alignment<13>::value == 16, "bad alignment calculated");
    static_assert(minimum_possible_alignment<14>::value == 16, "bad alignment calculated");
    static_assert(minimum_possible_alignment<15>::value == 16, "bad alignment calculated");
    static_assert(minimum_possible_alignment<16>::value == 16, "bad alignment calculated");
    static_assert(minimum_possible_alignment<17>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<18>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<19>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<20>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<21>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<22>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<23>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<24>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<25>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<26>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<27>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<28>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<29>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<30>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<31>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<32>::value == 32, "bad alignment calculated");
    static_assert(minimum_possible_alignment<33>::value == 64, "bad alignment calculated");

    align_by_type<char>(128);
    align_by_type<int>(128);
    align_by_type<double>(128);

    align_by_type<threepack<char>>(128);
    align_by_type<threepack<int>>(128);
    align_by_type<threepack<double>>(128);

    align_by_type<fourpack<char>>(128);
    align_by_type<fourpack<int>>(128);
    align_by_type<fourpack<double>>(128);

    align_by_type<fourpack<long long>>(128);

    // uncommet these to trigger compile time assertions
    //aligned_malloc<char, 3>(200);     // alignment is not multiple of sizeof(void*)
    //aligned_malloc<char, 24>(200);    // alignment is not power of two

    return 0;
}
