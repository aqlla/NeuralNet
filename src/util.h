// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_UTILITY_H
#define ML_UTILITY_H

#include <vector>
#include <sstream>
#include <iostream>
#include "itlib.h"

namespace it {
    namespace func {

        template<typename T>
        struct Range : std::pair<T, T>
        {
            constexpr Range(T f, T s)
                    : std::pair<T, T>(f, s) { };

            std::string to_string() const {
                std::stringstream ss;
                ss << "[" << this->first << ", " << this->second << "]";
                return ss.str();
            }
        };

        template<typename T>
        inline std::ostream &operator<<(std::ostream &out, const it::func::Range <T> &rhs) {
            return out << (rhs.to_string());
        }

    }
}


using std::unique_ptr;
using vec_f = std::vector<f64>;
using range = it::func::Range<f64>;

#endif //ML_UTILITY_H
