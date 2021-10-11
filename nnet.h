#include <iostream>
#include <string>
#include "layer.h"

#pragma once

using namespace std;

class NNet{
    public:

        double** trainingInputs;
        double** trainingOutputs;
        double* expectedOutput;
        int trainingSetsLoaded;
        bool sumWeightGradients = false;
        int curTrainingSet;

        double learningRate;

        string errorFunc;
        int layerCount;

        Layer* inputLayer;
        Layer* outputLayer;

        double* curSetNodeError;
        double sumSqError;
      
        // Net functions
        NNet();
        void loadNetFromFile(string path);
        void loadTrainingFromFile(string path);
        void loadCurTrainingSet();
        void forwardProp();
        void backProp();
        void clearGradients();
        void trainOverSet(int epochs, bool sumWeightGradients);
        void applyWeightGradients();
        void addLayer(int num, int numNodes, string actvFunc);
        void randomizeWeights(double min, double max);
        void compError();
        void printNetwork();

        //void addLayer(int laye);

     


};