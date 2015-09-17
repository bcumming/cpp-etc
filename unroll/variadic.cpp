#include <iostream>

// the ct in ct_store and ct_array stands for "compile time"

template <unsigned index, unsigned ...args>
struct ct_store {
    static constexpr
    unsigned get(unsigned i) {
        return 0;
    }
};

template <unsigned index, unsigned value, unsigned ...args>
struct ct_store<index, value, args...> : ct_store<index-1, args...> {
    using head = ct_store<index-1, args...>;

    static constexpr
    unsigned get(unsigned i) {
        return i==index ? value : head::get(i);
    }
};

template <unsigned ...args>
struct ct_array {
    static constexpr
    unsigned size() {
        return sizeof...(args);
    }

    template<unsigned i>
    static constexpr
    unsigned get() {
        static_assert(i<size(), "out of bounds access");
        return values::get(size()-i);
    }

    using values = ct_store<size(), args...>;

    constexpr
    unsigned operator [] (unsigned i) const {
        return values::get(size()-i);
    }
};

template<typename X>
int dot(int *y) {
    X x;
    int sum = 0;
    #pragma unroll
    for(int i=0; i<X::size(); ++i) {
        sum += y[i]*x[i];
    }
    return sum;
}

using dot_table = ct_array<0, 0, 1, 2, 4, 4, 5, 6, 3, 1, 2, 3, 7, 5, 6, 7, 8, 9, 11, 10, 8, 9, 11, 10>;
template<int N>
int dot_template(int *y) {
    return dot_table::template get<N>()*y[0] + dot_template<N-1>(y+1);
}

template<>
int dot_template<0>(int *y) {
    return dot_table::template get<0>();
}

void demo_simple() {
    using ct = ct_array<1,2,3,5,7,11,13>;
    ct v;

    std::cout << "--- tuple style" << std::endl;

    std::cout << ct::get<0>() << std::endl;
    std::cout << ct::get<1>() << std::endl;
    std::cout << ct::get<2>() << std::endl;
    std::cout << ct::get<3>() << std::endl;
    std::cout << ct::get<4>() << std::endl;
    std::cout << ct::get<5>() << std::endl;
    std::cout << ct::get<6>() << std::endl;
    // uncomment this to test compile time bounds checking
    //std::cout << ct::get<7>() << std::endl;

    std::cout << "--- array style" << std::endl;


    for(auto i=0; i<v.size(); ++i) {
        std::cout << v[i] << std::endl;
    }
}

int demo_dot_template() {
    constexpr unsigned N = dot_table::size();
    int y[N];
    return dot_template<N-1>(y);
}

int demo_dot() {
    using table = ct_array<1,2,3,4,5,6,7,8>;
    constexpr unsigned N = table::size();
    int y[N] = {1, 2, 2, 1, 2, 1, 2, 1};

    return dot<table>(&y[0]);
}

int main(void) {
    demo_simple();
    //return demo_dot();
    return demo_dot_template();
}

