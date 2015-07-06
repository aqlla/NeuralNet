// Created by Aquilla Sherrock on 7/6/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#include "layer.hpp"

LayerBase::LayerBase(LayerBase&& other)
    : neurons{std::move(other.neurons)},
      neuronCount{other.neuronCount},
      prev{std::move(other.prev)} {};

LayerBase::LayerBase(const size_t neuronCount)
        : neuronCount{neuronCount},
          prev{nullptr} {};

LayerBase::LayerBase(const size_t neuronCount, const shared_ptr<LayerBase> prev)
        : neuronCount{neuronCount},
          prev{prev} {};

auto LayerBase::operator =(LayerBase&& other) -> LayerBase& {
    if (this != &other) {
        neurons = std::move(other.neurons);
        neuronCount = other.neuronCount;
        prev = std::move(other.prev);
    }

    return *this;
};


/**
 * Get NeuronBase Output Pointer.
 */
auto LayerBase::getOutputPtr(const size_t index) const -> const shared_ptr<f64> {
    return neurons[index]->output;
};

/**
 * Get NeuronBase Output Value.
 */
auto LayerBase::getNeuronValue(const size_t index) const -> const f64 {
    return *neurons[index]->output;
};

/**
 * Forward Propagate.
 */
void LayerBase::forward() {
    for (auto &n : neurons) {
        n->setOutput();
        #if NN_DEBUG == NN_DBG_ALL
        std::cout << *n << std::endl;
        #endif
    }
};

auto LayerBase::operator[](const size_t index) -> const f64 {
    return getNeuronValue(index);
};

auto LayerBase::size() const -> const size_t {
    return neuronCount;
};

