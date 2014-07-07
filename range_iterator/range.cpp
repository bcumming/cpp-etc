#include <iostream>

#include "range.h"

void test(size_t range_size, size_t num_splits) {
    std::cout << "--------------------------------------------------------" << std::endl;
    Range range(0, range_size);
    std::cout << "split range " << SplitRange(range, num_splits) << std::endl;
    for(auto subrange : SplitRange(range, num_splits))
        std::cout << subrange << " ";
    std::cout << std::endl;
}

int main(void) {
    size_t R[] = {5, 10};
    size_t S[] = {1, 2, 4, 5, 10};
    for(auto r : R)
        for(auto s : S)
            test(r,s);
}

