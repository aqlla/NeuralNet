#ifndef ML_NEURON_H
#define ML_NEURON_H

#include <itlib.h>
#include "synapse.hpp"

class NeuronBase {
public:
    NeuronBase();

    auto sumInputs()                        -> const f64;
    auto addInput(shared_ptr<f64>)          -> NeuronBase&;
    auto addInput(shared_ptr<f64>, f64)     -> NeuronBase&;

    virtual auto setOutput()                -> NeuronBase&;
    virtual auto calculateGrad(const f64)   -> const f64;
    virtual auto to_string() const          -> const std::string;

    f64 localGradient;
    shared_ptr<f64> output;

protected:
    f64 inputTotal;
    vector<Synapse> inputs;
};


template <class ActivationFunc>
class Neuron : public NeuronBase {
public:
    auto setOutput() -> Neuron& override {
        *output = activationFunc.f(sumInputs());
        return *this;
    };

    auto calculateGrad(const f64 desired) -> const f64 override {
        localGradient = activationFunc.df(inputTotal) * (*output - desired);
        return localGradient;
    };

    auto to_string() const -> const std::string override {
        std::stringstream ss;
        ss << NeuronBase::to_string()
           << "f(" << inputTotal << ")  = " << activationFunc.f(inputTotal) << std::endl
           << "df(" << inputTotal << ") = " << activationFunc.df(inputTotal) << std::endl;
        return ss.str();
    };

protected:
    ActivationFunc activationFunc;
};


inline auto operator <<(std::ostream &out, const NeuronBase &n) -> std::ostream& {
    return out << (n.to_string());
}


#endif //ML_NEURON_H