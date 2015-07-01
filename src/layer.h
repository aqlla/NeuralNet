// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_LAYER_H
#define ML_LAYER_H

#include "util.h"
#include "activation_functions.h"
#include "neuron.h"


using std::make_unique;
using std::vector;


class LayerBase {
public:
    LayerBase() = default;
    virtual ~LayerBase() = default;

    LayerBase(LayerBase &&other)
            : neurons{std::move(other.neurons)},
              neuronCount{other.neuronCount},
              prev{other.prev} {};

    LayerBase(size_t neuronCount)
            : neuronCount{neuronCount},
              prev{nullptr} {};

    LayerBase(size_t neuronCount, LayerBase *prev)
            : neuronCount{neuronCount},
              prev{nullptr} {};


    // Explicitly delete copy ctor/assignment op to protect unique_ptr
    LayerBase(LayerBase const &) = delete;
    LayerBase&operator =(LayerBase const &) = delete;

    LayerBase &operator =(LayerBase &&other) {
        if (this != &other) {
            neurons = std::move(other.neurons);
            neuronCount = other.neuronCount;
            prev = other.prev;
        }

        return *this;
    };

    virtual f64 *getOutputPtrAtIndex(size_t index) const = 0;
    virtual void forward() = 0;

    size_t neuronCount;

protected:
    LayerBase *prev;
    std::vector<unique_ptr<NeuronBase>> neurons;
};


template <size_t NeuronCount, class ActivationFunc = activation::_default>
class FullyConnectedLayer : public LayerBase
{
public:
    FullyConnectedLayer() = default;

    FullyConnectedLayer(LayerBase *prev)
            : LayerBase{NeuronCount, prev}
    {
        for (int i = 0; i < NeuronCount; ++i) {
            auto neuron = make_unique<Neuron<ActivationFunc>>();
            for (size_t j = 0; j < prev->neuronCount; ++j)
                neuron->addInput(prev->getOutputPtrAtIndex(j));
            neurons.push_back(std::move(neuron));
        }
    };

    f64 *getOutputPtrAtIndex(size_t index) const {
        return &neurons[index]->output;
    };

    void forward() override {
        for (auto &n : neurons) {
            n->setOutput();
            std::cout << *n << std::endl << std::endl;
        }
    };
};


template <size_t NeuronCount, class ActivationFunc = activation::identity>
class InputLayer : public LayerBase
{
protected:
    std::vector<f64*> inputs;

public:
    InputLayer() = default;

    InputLayer(std::vector<f64*> &inputs)
            : LayerBase{NeuronCount},
              inputs{inputs} {};

    f64 *getOutputPtrAtIndex(size_t index) const override {
        return inputs[index];
    };

    void forward() override {
        int i = 0;
        for (auto &n : inputs) {
            std::cout << "Input " << i++ << ": " << *n << std::endl;
        }
    };
};

#endif //ML_LAYER_H
