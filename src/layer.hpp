// Created by Aquilla Sherrock on 7/6/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_LAYER_H
#define ML_LAYER_H

#include "util.h"
#include "activation_functions.hpp"
#include "neuron.hpp"

class LayerBase {
public:
    LayerBase() = default;
    LayerBase(LayerBase&&);
    LayerBase(const size_t);
    LayerBase(const size_t, const shared_ptr<LayerBase>);
    virtual ~LayerBase() = default;

    // Explicitly delete copy ctor/assignment operator to protect unique_ptr
    LayerBase(LayerBase const&) = delete;
    LayerBase &operator =(LayerBase const&) = delete;
    LayerBase &operator =(LayerBase&&);

    /** Get NeuronBase Output Pointer. */
    virtual auto getOutputPtr(const size_t) const -> const shared_ptr<f64>;

    /** Get NeuronBase Output Value. */
    virtual auto getNeuronValue(const size_t) const -> const f64;

    /** Forward Propagate. */
    virtual void forward();

    /** Back Propogation */
    virtual void calculateGradients(const f64 desired, const shared_ptr<LayerBase> next) {
        for (auto& n : neurons) {
            n->calculateGrad(desired);
        }
    };

    virtual auto operator[](const size_t) -> const f64;
    virtual auto size() const -> const size_t;

    vector<unique_ptr<NeuronBase>> neurons;
protected:
    size_t neuronCount;
    shared_ptr<LayerBase> prev;
};


class InputLayer : public LayerBase
{
public:
    InputLayer() = default;
    InputLayer(const size_t neuronCount)
            : LayerBase{neuronCount} {};
    InputLayer(const size_t neuronCount, const vector<shared_ptr<f64>> &inputs)
            : LayerBase{neuronCount} {
        for (size_t i = 0; i < neuronCount; ++i) {
            auto neuron = make_unique<Neuron<activation::identity>>();
            neuron->addInput(inputs[i], Synapse::INPUT_WEIGHT);
            neurons.push_back(std::move(neuron));
        }
    };
};

#endif //ML_LAYER_H
