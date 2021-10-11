#include <iostream>
#include <string>
#include <math.h>
#include "nnet.h"
#include "layer.h"

using namespace std;


// Default constructor
Layer::Layer(int layerNum, int numNodes, string actvFunc){
    this->num = layerNum;
    this->numNodes = numNodes;
    this->actvFunc = actvFunc;

    
    bias= 0;
    input = NULL;
    output = NULL;
    weights = NULL;
    actvFunc = "";
    nextLayer = NULL;
    prevLayer = NULL;

    gradOutput = NULL;
    gradInput = NULL;

    gradWeights = NULL;
    

}

void Layer::procActvFunc(){
    for(int i = 0; i < numNodes; i++){
        if(actvFunc == "Sigmoid"){
            output[i] =  sigmoid(input[i]);
        }

    }
}

double Layer::sigmoid(double x){
    double result = 0;

    result =  1 / (1 + exp(-x));

    return result;

}

double Layer::sigmoidDeriv(double x){
    double result = 0;
    double sig_x = sigmoid(x);
    result =  sig_x*(1 - sig_x);

    return result;

}
