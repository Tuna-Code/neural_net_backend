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
        int curTrainingSet;

        double learningRate;

        string errorFunc;
        int layerCount;

        Layer* inputLayer;
        Layer* outputLayer;

        double* curSetNodeError;
        
        double sumSqrError;
        double crossEntropyError;
        double mseError;
      
        // Net functions
        NNet();
        void loadNetFromFile(string path);
        void loadTrainingFromFile(string path);
        void loadCurTrainingSet();
        void forwardProp();
        void backProp();
        void clearGradients();
        void trainOverSet(int epochs);
        void trainOverSetBatch(int epochs, int batchSize);
        void applyWeightGradients();
        void applyWeightGradientsBatch(int batchSize);
        void addLayer(int num, int numNodes, string actvFunc);
        void randomizeWeights(double min, double max);
        void compError();
        void printNetwork();
        void printNetworkError();

        //void addLayer(int laye);

     


};