// Created by Aquilla Sherrock on 7/6/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_SYNAPSE_H
#define ML_SYNAPSE_H

#include <itlib.h>
#include "util.h"

class Synapse {
public:
    Synapse(shared_ptr<f64>);
    Synapse(shared_ptr<f64>, f64);

    auto get() const -> const f64;
    static auto randomWeight() -> const f64;
    auto to_string() const -> const std::string;

    f64 weight;
    f64 deltaWeight;
    shared_ptr<f64> input;
    static constexpr f64 INPUT_WEIGHT = 1.0;
};


#endif //ML_SYNAPSE_H
