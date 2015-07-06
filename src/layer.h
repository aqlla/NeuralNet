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

    LayerBase(const size_t neuronCount)
            : neuronCount{neuronCount},
              prev{nullptr} {};

    LayerBase(const size_t neuronCount, const shared_ptr<LayerBase> prev)
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


    /**
     * Get Neuron Output Pointer.
     */
    virtual auto getOutputPtr(const size_t index) const -> const shared_ptr<f64> {
        return neurons[index]->output;
    };

    /**
     * Get Neuron Output Value.
     */
    virtual auto getNeuronValue(const size_t index) const -> const f64 {
        return *neurons[index]->output;
    };

    /**
     * Forward Propagate.
     */
    virtual void forward() {
        for (auto &n : neurons) {
            n->setOutput();
            std::cout << *n << std::endl;
        }
    };

    virtual auto operator[](const size_t index) -> const f64 {
        return getNeuronValue(index);
    };

    virtual auto size() const -> const size_t {
        return neuronCount;
    };

protected:
    size_t neuronCount;
    shared_ptr<LayerBase> prev;
    vector<unique_ptr<NeuronBase>> neurons;
};


template <class ActivationFunc>
class FullyConnectedLayer : public LayerBase
{
public:
    FullyConnectedLayer() = default;
    FullyConnectedLayer(const size_t neuronCount)
            : LayerBase{neuronCount} {};
    FullyConnectedLayer(const size_t neuronCount, const shared_ptr<LayerBase> prev)
            : LayerBase{neuronCount, prev} {
        for (size_t i = 0; i < neuronCount; ++i) {
            auto neuron = make_unique<Neuron<ActivationFunc>>();
            for (size_t j = 0; j < prev->size(); ++j)
                neuron->addInput(prev->getOutputPtr(j));
            neurons.push_back(std::move(neuron));
        }
    };

    virtual void forward() override {
        std::cout << "Hidden: " << std::endl;
        LayerBase::forward();
    };
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

template <class ActivationFunc>
class OutputLayer : public FullyConnectedLayer<ActivationFunc>
{
public:
    OutputLayer() = default;
    OutputLayer(const size_t neuronCount)
            : FullyConnectedLayer<ActivationFunc>{neuronCount} {};
    OutputLayer(const size_t neuronCount, const shared_ptr<LayerBase> prev)
            : FullyConnectedLayer<ActivationFunc>{neuronCount, prev} {};

    void forward() override {
        std::cout << "Output: " << std::endl;
        LayerBase::forward();
    };
};

#endif //ML_LAYER_H
