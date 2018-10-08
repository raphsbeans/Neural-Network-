#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

using namespace std;

//This is a connection
struct Connection
{
    double weight;
    double deltaWeight;
};

//This will be our Neuron Class
class Neuron;

//Let's create a Layer
typedef vector <Neuron> Layer;

//**************** Neuron ****************
class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal (double val) {m_outputVal = val;}
    double getOutputVal (void) const {return m_outputVal; }
    void feedForward (const Layer &prevLayer);
    void calcOutputGradients (double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights (Layer &prevLayer);

private:
    ///TODO: The weight's priors might need to be changed
    static double eta; // Overall net Learning rate
    static double alpha; // Multiplier of last weight change (momentum)
    static double randomWeight (void) {return rand ()/ double(RAND_MAX);}
    double sumDOW (const Layer &nextLayer) const;
    static double transferFunction (double x);
    static double transferFunctionDerivative (double x);
    double m_outputVal;
    vector <Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
};

///TODO: See if this values can be trust
double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

void Neuron::updateInputWeights (Layer &prevLayer)
{
    //The weights to be updated are in Connection container
    //in the neurons in the preceding layer
    for (unsigned n = 0; n< prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                //Individual input, magnified by the gradient and train rate:
                eta * neuron.getOutputVal() * m_gradient
                //Also add momentum = a fraction of the previous data weight
                + alpha * oldDeltaWeight;
        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    }
}

double Neuron::sumDOW (const Layer &nextLayer) const
{
    double sum = 0.0;

    //Sum our contributions of the errors at the nodes we feed

    for (unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW (nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients (double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x)
{
    //Tanh - output range [-1.0, 1.0]
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x)
{
    //A good aproximation could be 1 - x*x
    return 1.0 - tanh(x)*tanh(x);
}

void Neuron::feedForward (const Layer &prevLayer){
    double sum = 0.0;

    for (unsigned n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    ///TODO: Transfer Function is our activation function
    m_outputVal = Neuron::transferFunction (sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for (unsigned c = 0; c < numOutputs; ++c){
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    }
    m_myIndex = myIndex;
}
//**************** Net ********************

class Net{
/**
    This claas is responsable to create the neural network
**/
public:
    Net(const vector<unsigned> &topology);
    void feedForward (const vector<double> &inputVals);
    void backProp (const vector<double> &targetVals) ;
    void getResults (vector<double> &resultVals) const;

private:
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmothingFactor;
};

void Net::getResults (vector<double> &resultVals) const
{
    resultVals.clear();

    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n ){
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}

void Net::backProp (const vector<double> &targetVals)
{
    //Calculate overall net error - RMS
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;

    for (unsigned n = 0; n < outputLayer.size() - 1; ++n){
        //Not counting the bias
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta*delta;
    }
    m_error /= outputLayer.size() - 1; //get average error squared
    m_error = sqrt(m_error); //RMS

    //Implement a recent average measurement
    m_recentAverageError =
                (m_recentAverageError * m_recentAverageSmothingFactor + m_error)/
                (m_recentAverageSmothingFactor + 1);

    //Calculate output layer gradients
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n){
        //Not counting the bias
        outputLayer[n].calcOutputGradients (targetVals[n]);
    }

    //Calculate gradients on hidden layers
    for (unsigned layerNum = m_layers.size() - 2; layerNum > 0; --layerNum){
        Layer &hiddenLayer = m_layers[layerNum];
        Layer &nextLayer = m_layers [layerNum + 1];

        for (unsigned n = 0; n< hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    //For all layers from outputs to first hidden layer, we need to update connection weights
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n< layer.size() - 1; n++){
            layer[n].updateInputWeights(prevLayer);
        }
    }
}
void Net::feedForward(const vector<double> &inputVals)
{
    assert(inputVals.size() == m_layers[0].size() - 1);

    //Assign (latch) the input values into the input neurons
    for (unsigned i =0; i< inputVals.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);
    }

    //Forward Propagation
    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum -1];
        for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
            m_layers[layerNum][n].feedForward(prevLayer);
        }
    }
}

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layernum =0; layernum < numLayers; ++layernum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layernum == topology.size() - 1 ? 0 : topology[layernum - 1];

        //We have a new layer, now fill it with neurons,
        //and add a biaa neuron in each layer
        for (unsigned neuroNum = 0; neuroNum <= topology[layernum]; ++neuroNum){
            m_layers.back().push_back(Neuron(numOutputs, neuroNum));
            cout<<"WE MADE A NEURON! YEAH! Layer: " <<  layernum << " neuron: " << neuroNum << endl;
        }
        ///TODO: SEE IF THIS IS A GOOD SOLUTION
        //Force the bias node's output value to 1.0
        m_layers.back().back().setOutputVal(1.0);
    }
}

int main ()
{

    //A topology is a vector contaning how we are going to biuld our neural net
    //for instance a topology equals to [3, 2,1] will create a neural net with 3 layers
    //The fisrt will contain 3 neurons plus a bias one, the second one will have 2 neurons plus a bias
    //The last one will have then one neuron plus the bias
    vector <unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
    Net myNet (topology);

    vector<double> inputVals;
    myNet.feedForward(inputVals);

    vector<double> targetVals;
    myNet.backProp(targetVals);

    vector<double> resultsVals;
    myNet.getResults(resultsVals);
}
