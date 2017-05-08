/*******************************************************************
* Basic Feed Forward Neural Network Class
* ------------------------------------------------------------------
********************************************************************/

#ifndef NNetwork
#define NNetwork

#include "dataReader.h"

class neuralNetworkTrainer;

class neuralNetwork
{
	//class members
	//--------------------------------------------------------------------------------------------
private:

	//number of neurons
	int nInput, nHidden, nOutput, batchSize;
	
	//neurons
	double* inputNeurons;
	double* hiddenNeurons;
	double* outputNeurons;

	//weights
	double** wInputHidden;
	double** wHiddenOutput;

	double *device_output1;
	double *input;
	double *w1;

	double *device_output2;
	double *hidden;
	double *w2;
		
	//Friends
	//--------------------------------------------------------------------------------------------
	friend neuralNetworkTrainer;
	
	//public methods
	//--------------------------------------------------------------------------------------------

public:

	//constructor & destructor
	neuralNetwork(int numInput, int numHidden, int numOutput, int batchSize);
	~neuralNetwork();

	//weight operations
	bool saveWeights(char* outputFilename);
	double getSetAccuracy( std::vector<dataEntry*>& set );

	void printCudaInfo();

	//private methods
	//--------------------------------------------------------------------------------------------

private: 

	void initializeWeights();
	inline double activationFunction( double x );
	void feedForward( double* pattern );
	void feedForwardBatch(std::vector<double*> patternVector);
	
};

#endif
