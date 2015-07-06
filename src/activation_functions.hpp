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
    };

    struct identity : public function
    {
        f64 f(const f64 value) const override {
            return value;
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

        f64 df(const f64 y) const override {
            return f(y) * (1.0 - f(y));
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

        f64 df(const f64 y) const override {
            return 1.0 - y*y;
        };

        const ::range range() const {
            return ::range{-1.0, 1.0};
        };
    };

    using _default = identity;
};

#endif //ML_ACTIVATION_FUNCTIONS_H
