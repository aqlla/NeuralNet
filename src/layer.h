// Created by Aquilla Sherrock on 6/25/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_LAYER_H
#define ML_LAYER_H

#include "util.h"
#include "activation_functions.h"
#include "neuron.h"

template <class T>
using neuron_uptr = std::unique_ptr<Neuron<T>>;

template <class T>
using uptr = std::unique_ptr<T>;

class _BaseLayer
{
public:
    _BaseLayer() = default;
    virtual ~_BaseLayer() = default;

    std::vector<uptr<BaseNeuron>> neurons;
};

template <class ActivationFunc>
class Layer: virtual public _BaseLayer
{
public:
    Layer(size_t neuronCount, std::vector<f64*> &inputs)
            : neuronCount(neuronCount)
    {
        for (int i = 0; i < neuronCount; ++i) {
            neuron_uptr<ActivationFunc> neuron{ new Neuron<ActivationFunc> };
            neuron.get()->addInput(inputs[i]);
            neurons.push_back(std::move(neuron));
        }
    };

protected:
    const size_t neuronCount;
};


template <class ActivationFunc = activation::identity>
class InputLayer : public Layer<ActivationFunc>
{
public:
    InputLayer(size_t neuronCount, std::vector<f64*> &inputs)
            : LayerType(neuronCount, inputs) {};
protected:
    using LayerType  = Layer<ActivationFunc>;
};

#endif //ML_LAYER_H
