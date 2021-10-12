#include <iostream>
#include <string>

#pragma once

using namespace std;

class Layer{

    public:

        int num;
        int numNodes;
    
        double bias;    
        double* input;
        double* output;
        double** weights;
        double** gradWeights;
        string actvFunc;
        Layer* nextLayer; // Pointer to next layer in linked list
        Layer* prevLayer;

        double* gradOutput;
        double* gradInput;
        

        Layer(int layerNum, int numNodes, string actvFunc);
        void procActvFunc();

        double sigmoid(double input);
        double relu(double input);
        double sigmoidDeriv(double input);
};