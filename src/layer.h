// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_LAYER_H
#define ML_LAYER_H

#include "util.h"
#include "activation_functions.h"
#include "neuron.h"


class LayerBase {
public:
    LayerBase(LayerBase &&other)
            : neurons{std::move(other.neurons)},
              neuronCount{other.neuronCount},
              prev{std::move(other.prev)} {};

    LayerBase(size_t neuronCount)
            : neuronCount{neuronCount},
              prev{nullptr} {};

    LayerBase(size_t neuronCount, shared_ptr<LayerBase> prev)
            : neuronCount{neuronCount},
              prev{prev} {};

    LayerBase &operator =(LayerBase &&other) {
        if (this != &other) {
            neurons = std::move(other.neurons);
            neuronCount = other.neuronCount;
            prev = std::move(other.prev);
        }

        return *this;
    };

    LayerBase() = default;
    virtual ~LayerBase() = default;

    // Explicitly delete copy ctor/assignment operator to protect unique_ptr
    LayerBase(LayerBase const &) = delete;
    LayerBase &operator =(LayerBase const &) = delete;

    virtual shared_ptr<f64> getOutputPtr(size_t index) const = 0;
    virtual void forward() = 0;
    size_t neuronCount;

protected:
    shared_ptr<LayerBase> prev;
    vector<unique_ptr<NeuronBase>> neurons;
};


template <class ActivationFunc>
class FullyConnectedLayer : public LayerBase
{
public:
    FullyConnectedLayer() = default;
    FullyConnectedLayer(size_t neuronCount) : LayerBase{neuronCount} {};
    FullyConnectedLayer(size_t neuronCount, shared_ptr<LayerBase> prev)
            : LayerBase{neuronCount, prev}
    {
        for (size_t i = 0; i < neuronCount; ++i) {
            auto neuron = make_unique<Neuron<ActivationFunc>>();
            for (size_t j = 0; j < prev->neuronCount; ++j)
                neuron->addInput(prev->getOutputPtr(j));
            neurons.push_back(std::move(neuron));
        }
    };

    shared_ptr<f64> getOutputPtr(size_t index) const {
        return neurons[index]->output;
    };

    void forward() override {
        for (auto &n : neurons) {
            n->setOutput();
            std::cout << *n << std::endl << std::endl;
        }
    };
};

class InputLayer : public LayerBase
{
protected:
    vector<shared_ptr<f64>> inputs;

public:
    InputLayer() = default;
    InputLayer(size_t neuronCount) : LayerBase{neuronCount} {};
    InputLayer(size_t neuronCount, vector<shared_ptr<f64>> &inputs)
            : LayerBase{neuronCount},
              inputs{inputs} {};

    shared_ptr<f64> getOutputPtr(size_t index) const override {
        return inputs[index];
    };

    void forward() override {
        for (auto &in : inputs)
            std::cout << "input: " << *in << std::endl;
    };
};

//template <class ActivationFunc = activation::identity>
//class InputLayer : public LayerBase
//{
//protected:
//    vector<shared_ptr<f64>> inputs;
//
//public:
//    InputLayer() = default;
//    InputLayer(size_t neuronCount) : LayerBase{neuronCount} {};
//    InputLayer(size_t neuronCount, vector<shared_ptr<f64>> &inputs)
//            : LayerBase{neuronCount},
//              inputs{inputs} {};
//
//    shared_ptr<f64> getOutputPtr(size_t index) const override {
//        return inputs[index];
//    };
//
//    void forward() override {
//        for (auto &in : inputs)
//            std::cout << "input: " << *in << std::endl;
//    };
//};

#endif //ML_LAYER_H
