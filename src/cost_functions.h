// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.

#ifndef ML_LOSS_FUNCTIONS_H
#define ML_LOSS_FUNCTIONS_H

#include <cmath>
#include "util.h"

struct CostFunction {
    /**
     * Cost Function.
     * Finds the error given the correct value and the calculated value.
     *
     * @param y: the correct value.
     * @param y1: the output value.
     */
    virtual f64 f(const f64 y, const f64 y1) const = 0;

    /**
     * Derivative of the cost function.
     *
     * @param y: the correct value.
     * @param y1: the output value.
     */
    virtual f64 df(const f64 y, const f64 y1) const = 0;
};


// mean-squared-error function for regression.
struct mse: CostFunction {
    f64 f(const f64 y, const f64 y1) const override {
        return (y - y1) * (y - y1) / 2;
    };

    f64 df(const f64 y, const f64 y1) const override {
        return y - y1;
    };
};

// cross-entropy loss function for (multiple independent) binary classifications
struct cross_entropy: CostFunction {
    f64 f(const f64 y, const f64 y1) const override {
        return -y1 * std::log(y) - (1.0 - y1) * std::log(1.0 - y);
    };

    f64 df(const f64 y, const f64 y1) const override {
        return (y - y1) / (y * (1 - y));
    };
};

// cross-entropy loss function for multi-class classification
struct cross_entropy_multiclass: CostFunction {
    f64 f(const f64 y, const f64 y1) const override {
        return -y1 * std::log(y);
    };

    f64 df(const f64 y, const f64 y1) const override {
        return -y1 / y;
    };
};


// Do this with functors?
//struct LossFunction {
//    virtual f64 operator ()(f64 e, f64 r) const = 0;
//};
//
//struct LossFunctionDerivative {
//    virtual f64 operator ()(f64 e, f64 r) const = 0;
//};
//
//
//struct mse: LossFunction {
//    f64 operator ()(f64 y, f64 y1) const override {
//        return (y - y1) * (y - y1) / 2;
//    };
//
//    struct Derivative: LossFunctionDerivative {
//        f64 operator ()(f64 y, f64 y1) const override {
//            return y - y1;
//        };
//    } df;
//};

#endif //ML_LOSS_FUNCTIONS_H
