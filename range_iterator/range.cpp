#include <iostream>

#include "range.h"

void test(size_t range_size, size_t num_splits) {
    std::cout << "--------------------------------------------------------" << std::endl;
    Range ranges(0, range_size);
    SplitRange split(ranges,num_splits);
    std::cout << "split range " << split << std::endl;
    for(auto subrange : split)
        std::cout << subrange << " ";
    std::cout << std::endl;
}

int main(void) {
    size_t R[] = {5, 10, 20};
    size_t S[] = {1, 2, 3, 4, 5, 10, 25};
    for(auto r : R)
        for(auto s : S)
            test(r,s);
}

