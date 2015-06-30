// Created by Aquilla Sherrock on 6/20/15.
// Copyright (c) 2015 Insignificant Tech. All rights reserved.

#ifndef ML_NEURAL_NET_H
#define ML_NEURAL_NET_H

#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>

class Neuron;

typedef std::vector<unsigned> topology_t;
typedef std::vector<Neuron> Layer;

struct Connection
{
    double weight;
    double deltaWeight;
};


class Neuron {
public:
    /**
     * @param numOutputs: number of neurons in next layer.
     */
    Neuron(unsigned numOutputs, unsigned indexInLayer) {
        _indexInLayer = indexInLayer;

        for (int con = 0; con < numOutputs; ++con) {
            _outputWeights.push_back(Connection());
            _outputWeights.back().weight = randomWeight();
        }
    };

    void feedForward(const Layer &previousLayer) {
        double sum = 0;

        // sum prev layer's outputs including bias
        for (unsigned neuronIndex = 0; neuronIndex < previousLayer.size(); ++neuronIndex) {
            const Neuron &neuron = previousLayer[neuronIndex];
            sum += neuron.getOutputValue() * neuron._outputWeights[_indexInLayer].weight;
        }

        _outputValue = Neuron::transform(sum);
    };

    void setOutputValue(const double value) {
        _outputValue = value;
    };

    double getOutputValue() const {
        return _outputValue;
    };

    void calculateOutputGradients(const double targetValue) {
        double delta = targetValue - _outputValue;
        _gradient = delta * Neuron::transformDerivative(_outputValue);
    };

    void calculateHiddenGradients(const Layer &nextLayer) {
        double dow = sumDOW(nextLayer);
        _gradient = dow * Neuron::transform(_outputValue);
    };

    void updateInputWeights(Layer &previousLayer) {
        // the weights to be updated are in the previous layer

        for (unsigned neuronIndex = 0; neuronIndex < previousLayer.size(); ++neuronIndex) {
            const Neuron &neuron = previousLayer[neuronIndex];
            double oldDeltaWeight = neuron._outputWeights[_indexInLayer].deltaWeight;
            double newDeltaWeight =
                    // individual input magnified by the gradient and training rate
                    eta * neuron.getOutputValue() * _gradient
                    // also add momentum (a fraction of the previous delta weight)
                    + alpha * oldDeltaWeight;

            neuron._outputWeights[_indexInLayer].deltaWeight = newDeltaWeight;
            neuron._outputWeights[_indexInLayer].weight += newDeltaWeight;
        }
    };

private:
    constexpr static double eta   = 0.15;
    constexpr static double alpha = 0.5;

    double _outputValue;
    double _indexInLayer;
    double _gradient;
    std::vector<Connection> _outputWeights;

    double sumDOW(const Layer &layer) const {
        double sum = 0;

        // sum of contributions of the errors at the nodes we feed
        for (unsigned neuronIndex = 0; neuronIndex < layer.size() - 1; ++neuronIndex) {
            sum += _outputWeights[neuronIndex].weight * layer[neuronIndex]._gradient;
        }

        return sum;
    };

    static double transform(double x) {
        // tanh - output range [-1.0...1.0]
        return tanh(x);
    };

    static double transformDerivative(double x) {
        return 1.0 - x * x;
    };

    static double randomWeight() {
        return rand() / double(RAND_MAX);
    };
};


class Net {
public:
    Net(const topology_t &topology) {
        unsigned long numLayers = topology.size();

        // Create correct number of layers.
        for (unsigned layerIndex = 0; layerIndex < numLayers; ++layerIndex) {
            _layers.push_back(Layer());
            unsigned numOutputs = (layerIndex == numLayers - 1)
                                ? 0
                                : topology[layerIndex + 1];

            // Fill new layer with neurons and add a bias neuron to each layer.
            for (unsigned neuronIndex = 0; neuronIndex <= topology[layerIndex]; ++neuronIndex) {
                _layers.back().push_back(Neuron(numOutputs, neuronIndex));
                std::cout << "made neuron!" << std::endl;
            }
        }
    };

    void feedForward(const std::vector<double> &input) {
        // assert that input count == number of neurons minus the bias neuron
        assert(input.size() == _layers[0].size() - 1);

        // latch the input values into the input neurons.
        for (unsigned i = 0; i < input.size(); ++i) {
            _layers[0][i].setOutputValue(input[i]);
        }

        // forward propagation
        for (unsigned layerIndex = 1; layerIndex < _layers.size(); ++layerIndex) {
            Layer &previousLayer = _layers[layerIndex - 1];

            // loop thru all neurons in layer excluding the bias neuron
            for (unsigned neuron = 0; neuron < _layers[layerIndex].size() - 1; ++neuron) {
                _layers[layerIndex][neuron].feedForward(previousLayer);
            }
        }
    };

    void backProp(const std::vector<double> &targetValues) {
        // calculate overall Net error (RMS of output neuron errors)
        Layer &outputLayer = _layers.back();
        _error = 0;

        for (unsigned neuronIndex = 0; neuronIndex < outputLayer.size() - 1; ++neuronIndex) {
            double delta = targetValues[neuronIndex] - outputLayer[neuronIndex].getOutputValue();
            _error += delta * delta;
        }

        _error /= outputLayer.size() - 1;   // get square of error average
        _error = sqrt(_error);              // RMS


        // implement a recent average measurement:
        _recentAverageError = (_recentAverageError * _recentAverageSmoothingFactor + _error)
                / (_recentAverageSmoothingFactor + 1.0);


        // calculate output layer gradients
        for (unsigned neuronIndex = 0; neuronIndex < outputLayer.size() - 1; ++neuronIndex) {
            outputLayer[neuronIndex].calculateOutputGradients(targetValues[neuronIndex]);
        }

        // calculate gradients of hidden layers
        for (unsigned long layerIndex = _layers.size() - 2; layerIndex > 0; --layerIndex) {
            Layer &hiddenLayer = _layers[layerIndex];
            Layer &nextLayer = _layers[layerIndex + 1];

            for (unsigned neuronIndex = 0; neuronIndex < nextLayer.size(); ++neuronIndex) {
                hiddenLayer[neuronIndex].calculateHiddenGradients(nextLayer);
            }

        }

        // update all connection weights
        for (unsigned long layerIndex = _layers.size() - 1; layerIndex > 0; --layerIndex) {
            Layer &layer = _layers[layerIndex];
            Layer &previousLayer = _layers[layerIndex - 1];

            for (unsigned neuronIndex = 0; neuronIndex < layer.size(); ++neuronIndex) {
                layer[neuronIndex].updateInputWeights(previousLayer);
            }

        }
    };

    void getResults(std::vector<double> &result) const {
        result.clear();

        for (unsigned neuronIndex = 0; neuronIndex < _layers.back().size() - 1; ++neuronIndex) {
            result.push_back(_layers.back()[neuronIndex].getOutputValue());
        }
    };

private:
    std::vector<Layer> _layers;
    double _error;
    double _recentAverageError;
    double _recentAverageSmoothingFactor;
};



#endif //ML_NEURAL_NET_H
