// Created by Aquilla Sherrock on 6/26/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_NEURON_H
#define ML_NEURON_H

#include <iostream>
#include "util.h"
#include "neuralnet.h"

struct Synapse
{
    Synapse(shared_ptr<f64> input)
            : weight{randomWeight()},
              input{input} {};

    shared_ptr<f64> input;
    f64 weight;
    f64 deltaWeight;

    f64 get() const {
        return *input * weight;
    };

    static f64 randomWeight() {
        return static_cast<f64>(rand()) / static_cast<f64>(RAND_MAX);
    };
};


class NeuronBase {
public:
    NeuronBase()
            : output{make_unique<f64>(0)} {};

    virtual f64 sumInputs() = 0;
    virtual void addInput(shared_ptr<f64>) = 0;
    virtual void setOutput() = 0;
    virtual std::string to_string() const = 0;
    shared_ptr<f64> output;
};


template <class ActivationFunc>
class Neuron : public NeuronBase {
public:
    f64 sumInputs() {
        inputTotal = 0;
        for (auto &in : inputs)
            inputTotal += in.get();
        return inputTotal;
    };

    void addInput(shared_ptr<f64> in) {
        inputs.push_back(Synapse{in});
    };

    void setOutput() {
        sumInputs();
        *output = activationFunc.f(inputTotal);
    };

    std::string to_string() const {
        std::stringstream ss;

        for (int i = 0; i < inputs.size(); ++i) {
            ss << "Synapse " << i << " info:" << std::endl
               << "\tinput:  " << *inputs[i].input << std::endl
               << "\tweight: " << inputs[i].weight << std::endl
               << "\tsignal: " << inputs[i].get() << std::endl;
        }

        ss << "Input Total: " << inputTotal << std::endl
           << "f(" << inputTotal << ")  = " << activationFunc.f(inputTotal) << std::endl
           << "df(" << inputTotal << ") = " << activationFunc.df(inputTotal) << std::endl;

        return ss.str();
    };

protected:
    f64 inputTotal;
    vector<Synapse> inputs;
    ActivationFunc activationFunc;
};


inline std::ostream &operator <<(std::ostream &out, const NeuronBase &n) {
    return out << (n.to_string());
}


#endif //ML_NEURON_H
