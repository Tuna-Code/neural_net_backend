
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
	net->loadNetFromFile("data/very_simple.top");
	net->loadTrainingFromFile("data/very_simple.training");
	//net->randomizeWeights(1,1);
	
	net->forwardProp();
	


	//net->trainOverSet(200, true);	

	//net->curTrainingSet = 1;
	//net->forwardProp();	
	//net->loadCurTrainingSet();
	net->printNetwork();
	
	//Layer* layer = new Layer(0, 2, "Null");
	
	cout << "DONE!";

	return 0;
}
