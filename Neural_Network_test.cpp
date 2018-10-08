#include <vector>
#include <iostream>

using namespace std;

//This will be our Neuron Class
class Neuron {};

//Let's create a Layer
typedef vector <Neuron> Layer;

class Net{
/**
    This claas is responsable to create the neural network
**/
public:
    Net(const vector<unsigned> &topology);
    void feedForward (const vector<double> &inputVals){};
    void backProp (const vector<double> &targetVals) {};
    void getResults (vector<double> &resultVals) const {};

private:
    vector<Layer> m_layers; //m_layers[layerNum][neuronNum]
};

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for (unsigned layernum =0; layernum < numLayers; ++layernum)
    {
        m_layers.push_back(Layer());

        //We have a new layer, now fill it with neurons,
        //and add a biaa neuron in each layer
        for (unsigned neuroNum = 0; neuroNum <= topology[layernum]; ++neuroNum){
            m_layers.back().push_back(Neuron());
            cout<<"WE MADE A NEURON! YEAH! Layer: " <<  layernum << " neuron: " << neuroNum << endl;
        }
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
