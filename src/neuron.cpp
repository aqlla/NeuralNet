// Created by Aquilla Sherrock on 7/6/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#include "neuron.hpp"

NeuronBase::NeuronBase()
        : output{ make_unique<f64>(0) } {};

auto NeuronBase::sumInputs() -> const f64 {
    inputTotal = 0;
    for (auto &in : inputs)
        inputTotal += in.get();
    return inputTotal;
};

auto NeuronBase::addInput(shared_ptr<f64> in) -> NeuronBase& {
    inputs.push_back(Synapse{in});
    return *this;
};

auto NeuronBase::addInput(shared_ptr<f64> in, f64 weight) -> NeuronBase& {
    inputs.push_back(Synapse{in, weight});
    return *this;
};

auto NeuronBase::calculateGrad(const f64 desired) -> const f64 {
    return localGradient = 0.0;
};

auto NeuronBase::setOutput() -> NeuronBase& {
    return *this;
};

auto NeuronBase::to_string() const -> const std::string {
    std::stringstream ss;
    for (size_t i = 0; i < inputs.size(); ++i)
        ss << "Synapse " << i << ":\n" << inputs[i].to_string();
    ss << "Input Total: " << inputTotal << std::endl;
    return ss.str();
};