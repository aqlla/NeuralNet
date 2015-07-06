// Created by Aquilla Sherrock on 6/26/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_NEURON_H
#define ML_NEURON_H

#include <iostream>
#include "util.h"
#include "neuralnet.h"

class NeuronBase;

template <class ActivationFunc>
class Neuron;

struct Synapse {
    Synapse(shared_ptr<f64> input)
            : weight{randomWeight()},
              input{input} {};

    Synapse(shared_ptr<f64> input, f64 weight)
            : weight{weight},
              input{input} {};

    shared_ptr<f64> input;
    f64 weight;
    f64 deltaWeight;

    auto get() const -> const f64 {
        return *(input) * weight;
    };

    static constexpr f64 INPUT_WEIGHT = 1.0;
    static auto randomWeight() -> const f64 {
        return static_cast<f64>(rand()) / static_cast<f64>(RAND_MAX);
    };
};


class NeuronBase {
public:
    NeuronBase()
            : output{ make_unique<f64>(0) } {};

    virtual auto sumInputs() -> const f64 {
        inputTotal = 0;
        for (auto &in : inputs)
            inputTotal += in.get();
        return inputTotal;
    };

    virtual auto addInput(shared_ptr<f64> in) -> NeuronBase& {
        inputs.push_back(Synapse{in});
        return *this;
    };

    virtual auto addInput(shared_ptr<f64> in, f64 weight) -> NeuronBase& {
        inputs.push_back(Synapse{in, weight});
        return *this;
    };

    virtual auto setOutput() -> NeuronBase& {
        return *this;
    };

    virtual auto to_string() const -> const std::string {
        std::stringstream ss;

        for (size_t i = 0; i < inputs.size(); ++i) {
            ss << "Synapse " << i << ":" << std::endl
            << "\tinput:  " << *(inputs[i].input) << std::endl
            << "\tweight: " << inputs[i].weight   << std::endl
            << "\tsignal: " << inputs[i].get()    << std::endl;
        }

        return ss.str();
    };

    shared_ptr<f64> output;

protected:
    f64 inputTotal;
    vector<Synapse> inputs;
};


template <class ActivationFunc>
class Neuron : public NeuronBase {
public:
    auto setOutput() -> NeuronBase& override {
        *output = activationFunc.f(sumInputs());
        return *this;
    };

    auto to_string() const -> const std::string override {
        std::stringstream ss;
        ss << NeuronBase::to_string()
           << "Input Total: " << inputTotal << std::endl
           << "f(" << inputTotal << ")  = " << activationFunc.f(inputTotal) << std::endl
           << "df(" << inputTotal << ") = " << activationFunc.df(inputTotal) << std::endl;

        return ss.str();
    };

protected:
    ActivationFunc activationFunc;
};


inline auto operator <<(std::ostream &out, const NeuronBase &n) -> std::ostream& {
    return out << (n.to_string());
}


#endif //ML_NEURON_H
