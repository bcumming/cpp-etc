#pragma once

namespace gpu {
    template <typename T, std::size_t N>
    class array {
        public :

        using value_type = T;
        constexpr std::size_t size() const {
            return N;
        }

        __device__
        array()
        { }

        __device__
        array(std::initializer_list<value_type> values) {
            auto n = values.size() < N ? values.size() : N;
            auto v = values.begin();
            for(auto i=0; i<n; ++i, ++v) {
                data_[i] = *v;
            }
        }

        __device__
        array(T value) {
            #pragma unroll
            for(auto i=0; i<N; ++i) {
                data_[i] = value;
            }
        }

        __device__
        array(T values[N]) {
            #pragma unroll
            for(auto i=0; i<N; ++i) {
                data_[i] = values[i];
            }
        }

        __device__
        value_type* begin() {
            return data_;
        }

        __device__
        value_type* end() {
            return data_+N;
        }

        __device__
        const value_type* begin() const {
            return data_;
        }

        __device__
        const value_type* end() const {
            return data_+N;
        }

        __device__
        T& operator[] (size_t i) {
            return data_[i];
        }

        __device__
        T const& operator[] (size_t i) const {
            return data_[i];
        }

        private :

        T data_[N];
    };
} // namespace gpu
