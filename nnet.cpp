#include <iostream>
#include <string>
#include <math.h>
//#include "layer.h"
#include "nnet.h"
#include <fstream>
#include <random>
#include <chrono>


// Neural net constructor
NNet::NNet(){
    learningRate = 0;
    errorFunc = "";
    layerCount = 0;
    inputLayer = NULL;
    outputLayer = NULL;
    trainingInputs = NULL;
    trainingOutputs = NULL;
    expectedOutput = NULL;
    trainingSetsLoaded = 0;
    curTrainingSet = 0;
    sumWeightGradients = false;

    curSetNodeError = NULL;
    sumSqError = 0;
    crossEntropyError = 0;

}

void NNet::forwardProp(){
    Layer* temp = inputLayer->nextLayer;
    
    // Load current training data into our input layer and expected output array
    loadCurTrainingSet();
    
    // Loop through each layer
    while(temp != NULL){
        // Loop through each node in our layer
        for(int i = 0; i < temp->numNodes; i++){
            // Loop through each node in prev layer(and clear current value from prev prop since we're doing summation)
            temp->input[i] = 0;
           for(int j = 0; j < temp->prevLayer->numNodes; j++){
               // Current input of node i is previous node j * weight[j][i] in weight matrix
               temp->input[i] += (temp->prevLayer->output[j] * temp->prevLayer->weights[j][i]);
               temp->output[i] = temp->input[i];
           }
           temp->input[i] += temp->bias;
            // If not input layer, compute weight*input product matrix
           
        }
        temp->procActvFunc();
        temp = temp->nextLayer;
    }
    compError();
}

void NNet::backProp(){
    Layer* cur = outputLayer;

    
    while(cur != inputLayer){

        // Backprop for CE Error and various layers (very-simple site)
        // https://www.ics.uci.edu/~pjsadows/notes.pdf

        if(errorFunc == "CE"){

            if(cur->actvFunc == "Softmax"){
                
               
                
                // Compute gradient out/in
                // Compute denominator for softmax deriv
                for(int i = 0; i < cur->numNodes; i++){
                    cur->gradOutput[i] = -1.0*(expectedOutput[i]/cur->output[i]);
                    
                    if(i == cur->numNodes - 1){
                        cur->gradInput[i] = cur->output[i]*(1 - cur->output[i]);
                    }
                    else{
                        
                        cur->gradInput[i] = -1*(cur->output[i]*cur->output[cur->numNodes-1]);
                    }
                }
                




            }

        }




















     /// ---------------- Working simple backprop for sigmoid only and SOS error

        //cout <<"\n-----------\n" << "Layer: " << cur-> num << endl;
        // If current layer has sigmoud output
        if(cur->actvFunc == "Sigmoid" && errorFunc == "SOS"){
            // Loop through each node
            for(int i = 0; i < cur->numNodes; i++){
                // If output layer, gradient output is cimple deriv of error
                if(cur == outputLayer){
                    cur->gradOutput[i] = 2*curSetNodeError[i];
                }
                else{
                    // cur->gradOutput[i] = cur->nextLayer->gradInput[i];
                    for(int j = 0; j < cur->nextLayer->numNodes; j++){
                        cur->gradOutput[i] += cur->nextLayer->gradInput[j]*cur->weights[i][j];
                    }
                }
                // Compute gradient of output layer inputs
                
                cur->gradInput[i] = cur->gradOutput[i] * cur->sigmoidDeriv(cur->input[i]);
            }
            // Loop through prev layer weights and compute weight gradient based on current gradient in + prev layer outputs
            for(int i = 0; i < cur->prevLayer->numNodes; i++){
                for(int j = 0; j < cur->numNodes; j++){
                    if(sumWeightGradients){
                        cur->prevLayer->gradWeights[i][j] += cur->gradInput[j]*cur->prevLayer->output[i];
                    }
                    else{
                        cur->prevLayer->gradWeights[i][j] = cur->gradInput[j]*cur->prevLayer->output[i];
                    }
                   // printf("%f ", cur->prevLayer->gradWeights[i][j]);
                    //cout << cur->prevLayer->gradWeights[i][j] << " ";
                }
                //cout << endl;
            }
        }
        //cout << "\n---------\n";
        cur = cur->prevLayer;
    }
   

}

void NNet::trainOverSet(int epochs, bool sumGradients){
    sumWeightGradients = sumGradients;
  
    
    for(int i = 0; i < epochs; i++){
        curTrainingSet = 0;
        for(int j = 0; j < trainingSetsLoaded; j++){
            forwardProp();
            backProp();
            if(!sumGradients){
                applyWeightGradients();
            }
            curTrainingSet++;
        }
        if(sumGradients){
            applyWeightGradients();
        }
        clearGradients();
    }
    curTrainingSet = 0;
    forwardProp();
}

void NNet::clearGradients(){
    Layer* cur = inputLayer;

    while(cur != outputLayer){
        for(int i = 0; i < cur->numNodes; i++){
            cur->gradInput[i] = 0;
            cur->gradOutput[i] = 0;
            for(int j = 0; j < cur->nextLayer->numNodes; j++){
                cur->gradWeights[i][j] = 0;
            }
        }
        cur = cur->nextLayer;
    }
}
void NNet::applyWeightGradients(){
    Layer* cur = inputLayer;

    while(cur != outputLayer){
        for(int i = 0; i < cur->numNodes; i++){
            for(int j = 0; j < cur->nextLayer->numNodes; j++){
                cur->weights[i][j] += cur->gradWeights[i][j] * learningRate;
            }
        }
        cur = cur->nextLayer;
    }



}

void NNet::loadCurTrainingSet(){
    Layer* temp = inputLayer;

    for(int i = 0; i < temp->numNodes; i++){
        temp->input[i] = trainingInputs[curTrainingSet][i];
        temp->output[i] = temp->input[i];
    }
    temp = outputLayer;
    for(int i = 0; i < temp->numNodes; i++){
        expectedOutput[i] = trainingOutputs[curTrainingSet][i];
    }
}

void NNet::randomizeWeights(double min, double max){
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> range(min,max);

    Layer* temp = inputLayer;

    while(temp != outputLayer){
        for(int i = 0; i < temp->numNodes; i++){
            for(int j = 0; j < temp->nextLayer->numNodes; j++){
                temp->weights[i][j] = range(generator);
            }
        }
        temp = temp->nextLayer;
    }
}

void NNet::printNetwork(){
    Layer* temp = inputLayer;

    cout << "Network Error Func: " << errorFunc << endl;
    cout << "Network Learning Rate: " << learningRate << endl;
    cout << "Training Sets Loaded: " << trainingSetsLoaded << endl;
    cout << "Current Training Set: " << (curTrainingSet + 1) << endl << endl;

    for(int i = 0; i < layerCount; i++){
        printf("\n* Layer: %i\n* Nodes:%i\n", temp->num, temp->numNodes);
        cout << "* Actv Func: " << temp->actvFunc << endl << endl;
        
        cout << "** Inputs:\n------------\n";
        for(int j = 0; j < temp->numNodes; j++){
            cout << temp->input[j] << endl;
        }
        cout << endl;

        cout << "** Outputs:\n------------\n";
        for(int j = 0; j < temp->numNodes; j++){
            cout << temp->output[j] << endl;
        }
        cout << endl;
        
        
        
        
       
        if(temp != outputLayer){
            cout << "** Weights:\n------------\n";
            for(int i = 0; i < temp->numNodes; i++){
                for(int j = 0; j < temp->nextLayer->numNodes; j++){
                        cout << temp->weights[i][j] << " ";
                }
                cout << endl;
            }
        }
        else{
            cout << "********************\n";
            cout << "\n\n--------- Network Error Analysis -----------\n\nNetwork Outputs:\n-----\n";
            for(int j = 0; j < temp->numNodes; j++){
                cout << temp->output[j] << endl;
            }
            cout << endl;
            cout << "Expected Outputs:\n------------\n";
            for(int j = 0; j < temp->numNodes; j++){
                cout << expectedOutput[j] << endl;
            }
            cout << endl;
            cout << "Network Error on Set:\n------------\n";
            for(int j = 0; j < temp->numNodes; j++){
                cout << curSetNodeError[j] << endl;
            }
            cout << "\nCross Entropy Error on Set:\n------------\n";
            cout << crossEntropyError << endl;
        }
        
        cout << "\n********************\n";
        temp = temp->nextLayer;
    }
}

void NNet::compError(){
    Layer* output = outputLayer;
    double curError = 0;

    double out = 0;
    double expOut = 0;
    double curEntropyError = 0;

    if(errorFunc == "SOS"){
        for(int i = 0; i < output->numNodes; i++){
            curError = expectedOutput[i] - output->output[i];
            curSetNodeError[i] = curError;
        }
    }
    else if(errorFunc == "CE"){
        for(int i = 0; i < output->numNodes; i++){
            curError = expectedOutput[i] - output->output[i];
            curSetNodeError[i] = curError;

            expOut = expectedOutput[i];
            out = output->output[i];
            
            //curEntropyError = (expOut*log10(out)) + ((1 - expOut)*log((1 - out)));
            
            //if(expOut != 0){
                //curEntropyError += expOut*log10(out);
                //if(expOut == 1){
                 //   curEntropyError += expOut*log10(out);
                    //printf("%f %f %f\n",expOut,out,log10(out));
               // }
               // else if(expOut == 0){
                 //   curEntropyError += log10(1.0 - out);
                   // printf("%f %f %f\n",expOut,out,log10(1.0 - out));
               // }
            curEntropyError += expOut*log10(out);
                
                
            //}

        }

        curEntropyError *= -1;
        //curEntropyError /= output->numNodes;
        crossEntropyError = curEntropyError;
    }



}
void NNet::addLayer(int num, int numNodes, string actvFunc){

    Layer* newLayer = new Layer(num, numNodes, actvFunc);


    // If this is our first layer added
    if(layerCount == 0){
        inputLayer = newLayer;
        outputLayer = newLayer;
        layerCount++;
    }
    else{
        outputLayer->nextLayer = newLayer;
        newLayer->prevLayer = outputLayer;
        outputLayer = newLayer;
        layerCount++;
    }
    newLayer->input = new double[numNodes];
    newLayer->output = new double[numNodes];

    newLayer->gradInput = new double[numNodes];
    newLayer->gradOutput = new double[numNodes];

    for(int i = 0; i < numNodes; i++){
        newLayer->input[i] = 0;
        newLayer->output[i] = 0;
        newLayer->gradInput[i] = 0;
        newLayer->gradOutput[i] = 0;
        
    }

    if(newLayer->num > 0){
        int rows = newLayer->prevLayer->numNodes;
        int cols = newLayer->numNodes;

        newLayer->prevLayer->weights = new double*[rows];
        newLayer->prevLayer->gradWeights = new double*[rows];
        
        for(int i = 0; i < rows; i++){
            newLayer->prevLayer->weights[i] = new double[cols];
            newLayer->prevLayer->gradWeights[i] = new double[cols];
            for(int j = 0; j < cols; j++){
                newLayer->prevLayer->weights[i][j] = 0;
                newLayer->prevLayer->gradWeights[i][j] = 0;

            }
        }

    }

}

void NNet::loadTrainingFromFile(string path){
    std::ifstream input(path);
    string curLine = "";
    trainingSetsLoaded = 0;
    int stringStart = 0;
    int stringEnd = 0;
    string temp_d = "";
    double temp = 0;
    int curCol = 0;
    int curRow = 0;
    

    if(input.is_open()){

        while(getline(input, curLine)){
            if(curLine.find("#") != string::npos || curLine.size() == 0){
				continue;
			}
			// If line is data, increment our counter
			else{
				trainingSetsLoaded++;
			}
        }
        input.close();
     }
    else{
        cout << "\nCANNOT OPEN FILE\n";
    }
    trainingInputs = new double*[trainingSetsLoaded];
    trainingOutputs = new double*[trainingSetsLoaded];
    
    for(int i = 0; i < trainingSetsLoaded; i++){
        trainingInputs[i] = new double[inputLayer->numNodes];
        trainingOutputs[i] = new double[outputLayer->numNodes];

    }

    
    input.open(path);
    if(input.is_open()){

        while(getline(input, curLine)){
            curCol = 0;
            if(curLine.find("#") != string::npos || curLine.size() == 0){
				continue;
			}
            else{
                stringStart = 0;
                				// Loop from starting position until EOL
				for(string::size_type i = stringStart; i < curLine.size(); i++){
						// If current char is space, ignore
                    if(curLine[i] == ' '){
                        continue;
                    }
                    // If on comma, grab preceeding text and convert to double and store
                    else if (curLine[i] == ','){
                        stringEnd = i;
                        temp_d = curLine.substr(stringStart,stringEnd-stringStart);
                        temp = stod(temp_d);
                        if(curCol < inputLayer->numNodes){
                            //printf("%i %i %f\n", curRow, curCol, temp);
                            trainingInputs[curRow][curCol] = temp;
                        }
                        else{
                            trainingOutputs[curRow][curCol - inputLayer->numNodes] = temp;
                        }
                        // Store value in our 2d array and increment position
                        //net->training_data[cur_row][cur_entry] = temp;
                       // cur_entry++;
                        stringStart = stringEnd + 1;
                        curCol++;
                    }
                    // If at end of line, grab remaining text (last value) and store
                    if(i == curLine.size() -1){
                        stringEnd = i;
                        temp_d = curLine.substr(stringStart,stringEnd);
                        temp = stod(temp_d);
                        if(curCol < inputLayer->numNodes){
                            trainingInputs[curRow][curCol] = temp;
                        }
                        else{
                            trainingOutputs[curRow][curCol - inputLayer->numNodes] = temp;
                        }

                        // Store value in our 2d array and increment position
                        //net->training_data[cur_row][cur_entry] = temp;
                       // cur_entry++;
                       curCol++;
                    }
                }
                curRow++;
            }
        }

    }
}

void NNet::loadNetFromFile(string path){
    int curInputPos = 0;
    string curLine = "";
    int desiredLayers = 0;
    std::ifstream input(path);
    int* nodesPerLayer = NULL;
    string* layerActvFunc = NULL;    
    int layerCounter = 0;
    int stringStart = 1;
    int stringEnd = 0;
    bool randWeights = false;
    double randMin = 0;
    double randMax = 0;

    bool newWeights = true;
    int wRow = 0;
    int wCol = 0;
    Layer* wTemp = NULL;
    double wVal = 0;
    if(input.is_open()){

        while(getline(input, curLine)){
           if(curLine.find("#") != string::npos || curLine.size() == 0){
				continue;
			}
            else{
                if(curInputPos == 0){
                    //cout << curLine << endl;
                    learningRate = stod(curLine);
                    curInputPos++;
                }   
                else if(curInputPos == 1){
                    errorFunc = curLine;
                    curInputPos++;

                }
                else if(curInputPos == 2){
                    desiredLayers = stoi(curLine);
                    nodesPerLayer = new int[desiredLayers];
                    layerActvFunc = new string[desiredLayers];
                    curInputPos++;

                }
                else if(curInputPos == 3){
                    layerCounter = 0;
                    for(string::size_type i = 0; i < curLine.size(); i++){
                        if(curLine[i] != '[' && curLine[i] != ']' && curLine[i] != ' ' && curLine[i] != ','){
                            nodesPerLayer[layerCounter] = (int) (curLine[i] - 48);
                            layerCounter++;
                        }
                    }
                    curInputPos++;
                }
                else if(curInputPos == 4){
                    layerCounter = 0;
                    for(string::size_type i = 1; i < curLine.size(); i++){
						if(curLine[i] == ',' || curLine[i] == ']'){
							stringEnd = i;
							layerActvFunc[layerCounter] = curLine.substr(stringStart,stringEnd-stringStart);
							layerCounter++;
							stringStart = stringEnd + 1;
						}
					}
                    curInputPos++;
                }
                else if(curInputPos == 5){
                    if(curLine == "1"){
                        randWeights = true;
                    }
                    else if(curLine == "0"){
                        randWeights = false;
                    }
                    curInputPos++;
                }

                else if(curInputPos == 6){
                    randMin = stod(curLine);
                    curInputPos++;

                }

                else if(curInputPos == 7){
                    randMax = stod(curLine);
                    curInputPos++;

                }
                else if(curInputPos == 8){
                    
                    if(inputLayer == NULL){
                        for(int i = 0; i < desiredLayers; i++){
                            addLayer(i, nodesPerLayer[i], layerActvFunc[i]);
                        }
                            curSetNodeError = new double[outputLayer->numNodes];
                            for(int j = 0; j < outputLayer->numNodes; j++){
                                 curSetNodeError[j] = 0;
                            }
                           
                        wTemp = inputLayer;
                    }
                    

                    
                    Layer* temp = outputLayer;
                    expectedOutput = new double[temp->numNodes];
                    for(int i = 0; i < temp->numNodes; i++){
                        expectedOutput[i] = 0;
                    }

                    // Init random weights
                    if(randWeights){
                        randomizeWeights(randMin, randMax);
                        return;
                    }
                    // Grab manual weight data
                    else{
                        if(newWeights){
                            stringStart = 1;
                            newWeights = false;

                        }
                        else{
                            stringStart = 0;
                        }

                        for(string::size_type i = 1; i < curLine.size(); i++){
                            


                            if(curLine[i] == ','){
                                stringEnd = i;
                                wVal = stod(curLine.substr(stringStart,stringEnd-stringStart));
                                wTemp->weights[wRow][wCol] = wVal;
                                wCol++;
                                stringStart = stringEnd + 1;

                            }
                            else if(curLine[i] == ']'){
                                newWeights = true;
                                stringEnd = i;
                                wVal = stod(curLine.substr(stringStart,stringEnd-stringStart));
                                wTemp->weights[wRow][wCol] = wVal;
                                wCol = 0;
                                wRow = 0;
                                stringStart = stringEnd + 1;
                                //cout << endl << endl;
                               
                                wTemp = wTemp->nextLayer;
                            }
                            else if(i == curLine.size() -1){
                                stringEnd = i + 1;
                                wVal =  stod(curLine.substr(stringStart,stringEnd-stringStart));
                                wTemp->weights[wRow][wCol] = wVal;
                                wCol = 0;
                                wRow++;
                                stringStart = stringEnd + 1;

                            }
                            
                           /* else if(i = curLine.size() - 1){
                                stringEnd = i;
							    cout << curLine.substr(stringStart,stringEnd-stringStart) << "\n" ;
                                stringStart = stringEnd + 1;
                            }*/
                            
                            
                        }
                        //cout << wLayer << " " << outputLayer->num <<  endl;
                        if(wTemp  == outputLayer){
                            return;
                        }
                       
                    }


                    
                }
                
            }
        }
       
        input.close();
        
    }
    else{
        cout << "\nCANNOT OPEN FILE\n";
    }
    // If we made it this far, setup manual weights
    
    
   

}
