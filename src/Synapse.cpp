// Created by Aquilla Sherrock on 7/6/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.

#include "synapse.hpp"

Synapse::Synapse(shared_ptr<f64> input)
        : weight{Synapse::randomWeight()},
          input{input} {};

Synapse::Synapse(shared_ptr<f64> input, f64 weight)
        : weight{weight},
          input{input} {};

auto Synapse::get() const -> const f64 {
    return *(input) * weight;
};

auto Synapse::randomWeight() -> const f64 {
    return static_cast<f64>(rand()) / static_cast<f64>(RAND_MAX);
};

auto Synapse::to_string() const -> const std::string {
    std::stringstream ss;
    ss << "\tinput:  " << *input << std::endl
       << "\tweight: " << weight << std::endl
       << "\tsignal: " << get()  << std::endl;
    return ss.str();
};