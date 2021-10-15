
//#include "utils.h"
#include "nnet.h"
#include "layer.h"
#include <iostream>

using namespace std;

int main()
{

	// ---------------------------------- Initialize Window Objs --------------------------------------

	// ---------------------------------- Initialize Neural Network Objs --------------------------------------

	// ---------------------------------- Initialize Graphics Objs --------------------------------------
	// Create our net and give it to our nehlper
	NNet* net = new NNet();
	net->loadNetFromFile("data/simple.top");
	net->loadTrainingFromFile("data/simple.training");
	//net->randomizeWeights(1,1);
	
	net->forwardProp();
	


	net->trainOverSet(200);	

	//net->curTrainingSet = 1;
	//net->forwardProp();	
	//net->loadCurTrainingSet();
//	net->outputLayer->output[0] = 0.2698;
	//net->outputLayer->output[1] = 0.3223;
//	net->outputLayer->output[2] = 0.4078;

	//net->outputLayer->input[0] = 1.8658;
	//net->outputLayer->input[1] = 2.2292;
	//net->outputLayer->input[2] = 2.8204;

	
	
	//net->compError();
	//net->crossEntropyError = 0.985;
	

	net->printNetwork();
	net->curTrainingSet = 1;
	net->forwardProp();
	
	net->printNetwork();
	//Layer* layer = new Layer(0, 2, "Null");
	
	cout << "DONE!";

	return 0;
}
