// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_ACTIVATION_FUNCTIONS_H
#define ML_ACTIVATION_FUNCTIONS_H

#include <cmath>
#include "util.h"

namespace activation 
{

    struct function
    {
        virtual const ::range range() const = 0;
        virtual f64 f(const f64) const = 0;
        virtual f64 df(const f64) const = 0;
        virtual f64 f(const vec_f &, size_t) const = 0;
        virtual vec_f df(const vec_f &y, size_t index) const {
            vec_f v(y.size(), 0);
            v[index] = df(y[index]);
            return v;
        };
    };

    struct identity : public function
    {
        f64 f(const f64 value) const override {
            return value;
        };

        f64 f(const vec_f &v, size_t index) const override {
            return v[index];
        };

        f64 df(const f64 y) const override {
            return 1;
        };

        const ::range range() const {
            return ::range{1.0, 0.0};
        };
    };


    struct sigmoid : public function
    {
        f64 f(const f64 value) const override {
            return 1.0 / (1.0 + std::exp(-value));
        };

        f64 f(const vec_f &v, size_t index) const override {
            return 1.0 / (1.0 + std::exp(-v[index]));
        };

        f64 df(const f64 y) const override {
            return y * (1.0 - y);
        };

        const ::range range() const {
            return ::range{1.0, 0.0};
        };
    };


    struct tanh : public function
    {
        f64 f(const f64 value) const override {
            return std::tanh(value);
        };

        f64 f(const vec_f &v, size_t index) const override {
            return std::tanh(v[index]);
        };

        f64 df(const f64 y) const override {
            return 1.0 - y*y;
        };

        const ::range range() const {
            return ::range{-1.0, 1.0};
        };
    };
};

#endif //ML_ACTIVATION_FUNCTIONS_H
