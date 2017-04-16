/*******************************************************************
* Neural Network Training Example
* ------------------------------------------------------------------
* Bobby Anguelov - takinginitiative.wordpress.com (2008)
* MSN & email: banguelov@cs.up.ac.za
*********************************************************************/

//standard libraries
#include <iostream>
#include <ctime>
#include <stdlib.h>

//custom includes
#include "neuralNetwork.h"
#include "neuralNetworkTrainer.h"

//use standard namespace
using namespace std;

int main()
{		
	//seed random number generator
	srand( (unsigned int) time(0) );
	
	//create data set reader and load data file
	dataReader d;
	d.loadDataFile("mnist_train.csv",784,10);
	d.setCreationApproach( STATIC, 10 );	

	//create neural network
	neuralNetwork nn(784,15,10);

	//create neural network trainer
	neuralNetworkTrainer nT( &nn );
	nT.setTrainingParameters(1.2, 0.9, false);
	nT.setStoppingConditions(100, 100);
	nT.enableLogging("log.csv", 5);
	
	//train neural network on data sets
	for (int i=0; i < d.getNumTrainingSets(); i++ )
	{
		nT.trainNetwork( d.getTrainingDataSet() );
	}

	//save the weights
	nn.saveWeights("weights.csv");
		
	cout << endl << endl << "-- END OF PROGRAM --" << endl;
	char c; cin >> c;
	return 0;
}
