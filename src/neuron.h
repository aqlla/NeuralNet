// Created by Aquilla Sherrock on 6/26/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_NEURON_H
#define ML_NEURON_H

#include <iostream>

struct Synapse {
    explicit Synapse() : weight(Synapse::randomWeight()) {
        set(0);
    };

    Synapse(f64 *input, f64 weight) : weight(weight) {
        this->input = input;
        set(*input);
    };

    f64 *input;
    f64 output;
    f64 weight;
    f64 deltaWeight;

    void set(const f64 value) {
        output = weight * value;
    };

    static f64 randomWeight() {
        return static_cast<f64>(rand()) / static_cast<f64>(RAND_MAX);
    };
};


class BaseNeuron
{
public:
    BaseNeuron() = default;
    virtual ~BaseNeuron() = default;

    f64 sumInputs() {
        inputTotal = 0;
        for (auto in : inputs)
            inputTotal += *in;
        return inputTotal;
    };

    void addInput(f64 *input) {
        inputs.push_back(input);
    };

    virtual void setOutput() = 0;

protected:
    f64 inputTotal;
    std::vector<f64*> inputs;
    std::vector<Synapse> outputs;
};


template <class ActivationFunc>
class Neuron : virtual public BaseNeuron
{
private:
    ActivationFunc activationFunc;

public:
    Neuron() {
//        std::cout << "neuron ctor called" << std::endl;
    };
//    ~Neuron() = default;


    virtual void setOutput() override {
        sumInputs();
        std::cout << "input sum: " << inputTotal << std::endl;
        std::cout << "f("<< inputTotal << "): " << activationFunc.f(inputTotal) << std::endl;

        activationFunc.f(inputTotal);

        for (auto &output : outputs)
            output.set(inputTotal);
    };

    std::string to_string() const {
        std::stringstream ss;
        ss << "Input Total: " << inputTotal << std::endl
            << "Inputs: " << std::endl;

        for (auto in : inputs) {
            ss << "Weight: " << in->weight << std::endl
            << "Signal: " << in->output << std::endl;
        }

        return ss.str();
    };
};


#endif //ML_NEURON_H
