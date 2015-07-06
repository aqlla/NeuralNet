// Created by Aquilla Sherrock on 7/6/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.


#ifndef ML_CONNECTED_LAYER_HPP
#define ML_CONNECTED_LAYER_HPP

#include "layer.hpp"

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
        #if NN_DEBUG == NN_DBG_ALL
        std::cout << "Hidden: " << std::endl;
        #endif

        LayerBase::forward();
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
        LayerBase::forward();
    };
};


#endif //ML_CONNECTED_LAYER_HPP
