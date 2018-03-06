//Created by Paul at Existor Ltd as part of a neural networks tutorial 1/9/2015
//See http://www.existor.com/en/news-neural-networks.html

//Libraries needed
#include <stdio.h> //for printf
#include <stdlib.h> //for rand
#include <string.h> //for memcpy
#include <math.h> //for tanh

//Number of training examples and nodes in each layer. Easier to define as compiler constants here
//than dynamically allocating memory at run-time.
#define NUM_TRAINING 3
#define NUM_INPUTS 3
#define NUM_HIDDEN 3
#define NUM_OUTPUTS 3

//Print out the weights of a vector of matrix. The vector/matrix is passed to this function as if it
//were a 1-dimensional array rather than 2 dimensions. In C, all the values are in one continuous
//block of memory so it doesn't really matter if we index it as a 1 or 2 dimensional array. But if
//we passed in 2 dimensional arrays, we'd need a different function for every possible 2D size.
void displayVectorOrMatrix (const char *label, float *m, int rows, int cols)
{
    int i;

	printf ("   %s:\n", label);
	for (i=0; i<rows*cols; i++)
        printf ("%10.5f%c", m[i], (cols>1 && i%cols==cols-1) ? '\n' : ' ');
	if (cols==1) printf ("\n");
}

//This is the main function in a C program.
//Compile and run this with: gcc neuralnetwork.c -o neuralnetwork.out; ./neuralnetwork.out
int main (int argc, char **argv)
{
	/// Set up inputs/outputs
	float inputs[NUM_TRAINING][NUM_INPUTS]   = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}}; //the 3 possible inputs ABC
	float outputs[NUM_TRAINING][NUM_OUTPUTS] = {{0, 1, 0}, {0, 0, 1}, {1, 0, 0}}; //the corresponding outputs BCA
	///float learningrate = 0.01; //learning rate
	float learningrate = 0.1; //learning rate
	int iterations = 100; //iterations
	int debug = 1; //0 for now debugging, 1 for the loss each iteration, 2 for all vectors/matrices each iteration

    if (argc > 2) {
        iterations = atoi(argv[1]);
        debug = atoi(argv[2]);
    }
	/// Initialise weights
	float Wxh[NUM_INPUTS+1][NUM_HIDDEN]; //weights between inputs x and hidden nodes, including an extra one for bias
	float Why[NUM_HIDDEN+1][NUM_OUTPUTS]; //weights between hidden nodes and output y, including an extra one for bias
	int t, i, h, o; //looping variables for iterations, input nodes, hidden nodes, output nodes

	for (i=0; i<NUM_INPUTS+1; i++)
        for (h=0; h<NUM_HIDDEN; h++)
            Wxh[i][h] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1;

	for (h=0; h<NUM_HIDDEN+1; h++)
        for (o=0; o<NUM_OUTPUTS; o++)
            Why[h][o] = ((float)rand() / (double)RAND_MAX) * 0.2 - 0.1;

    displayVectorOrMatrix ("input/hidden weights(Wxh)", (float*)Wxh, NUM_INPUTS+1, NUM_HIDDEN);
	displayVectorOrMatrix ("hidden/output weights(Why)", (float*)Why, NUM_HIDDEN+1, NUM_OUTPUTS);

	/// Variables for the training
	float x[NUM_INPUTS+1]; //input vector including one extra for bias
	float y[NUM_OUTPUTS]; //output vector
	float zhWeightedSums[NUM_HIDDEN]; //weighted sums for the hidden nodes
	float hActivationValues[NUM_HIDDEN+1]; //activation values of the hidden nodes, including one extra for the bias
	float zyWeightedSums[NUM_OUTPUTS]; //weighted sums for the output nodes
	float probabilities[NUM_OUTPUTS]; //activation values of the output nodes
	float outputErrors[NUM_OUTPUTS]; //error in the output
	float deltaWxh[NUM_INPUTS+1][NUM_HIDDEN]; //adjustments to weights between inputs x and hidden nodes
	float deltaWhy[NUM_HIDDEN+1][NUM_OUTPUTS]; //adjustments to weights between hidden nodes and output y
	float loss, sum; //for storing the loss

	/// Train the network
	for (t=0; t<iterations; t++)
	{
		/// Get input and output
		/// Fill the x and y array with the input and ouput
		if (debug>=2) printf ("--------------------------------Iteration %4d--------------------------------\n", t+1);
		x[0]=1;
		memcpy (&x[1], inputs[t % NUM_TRAINING], sizeof (inputs[0])); //copy the input into x with the bias=1
		if (debug>=2) displayVectorOrMatrix ("inputs", x, NUM_INPUTS+1, 1);
		memcpy (y, outputs[t % NUM_TRAINING], sizeof (outputs[0])); //copy the output into y
		if (debug>=2) displayVectorOrMatrix ("outputs", y, NUM_OUTPUTS, 1);

		///Forward propagation ------------------------------------------------
		//Start the forward pass by calculating the weighted sums and activation values for the hidden layer
		memset (zhWeightedSums, 0, sizeof (zhWeightedSums)); //set all the weighted sums to zero
		for (h=0; h<NUM_HIDDEN; h++)
            for (i=0; i<NUM_INPUTS+1; i++)
                zhWeightedSums[h] += x[i] * Wxh[i][h]; //multiply and sum inputs * weights
		if (debug>=2) displayVectorOrMatrix ("input/hidden weights", (float*)Wxh, NUM_INPUTS+1, NUM_HIDDEN);
		if (debug>=2) displayVectorOrMatrix ("hidden weighted sums", zhWeightedSums, NUM_HIDDEN, 1);

		hActivationValues[0]=1; //set the bias for the first hidden node to 1
		for (h=0; h<NUM_HIDDEN; h++)
            hActivationValues[h+1] = tanh (zhWeightedSums[h]); //apply activation function on other hidden nodes
		if (debug>=2) displayVectorOrMatrix ("hidden node activation values", hActivationValues, NUM_HIDDEN+1, 1);

		memset (zyWeightedSums, 0, sizeof (zyWeightedSums)); //set all the weighted sums to zero
		for (o=0; o<NUM_OUTPUTS; o++)
            for (h=0; h<NUM_HIDDEN+1; h++)
                zyWeightedSums[o] += hActivationValues[h] * Why[h][o]; //multiply and sum inputs * weights
		if (debug>=2) displayVectorOrMatrix ("hidden/output weights", (float*)Why, NUM_HIDDEN+1, NUM_OUTPUTS);
		if (debug>=2) displayVectorOrMatrix ("output weighted sums", zyWeightedSums, NUM_OUTPUTS, 1);

		for (sum=0, o=0; o<NUM_OUTPUTS; o++) {
            probabilities[o] = exp (zyWeightedSums[o]);
            sum += probabilities[o];
        } //compute exp(z) for softmax
		for (o=0; o<NUM_OUTPUTS; o++) probabilities[o] /= sum; //apply softmax by dividing by the the sum all the exps
		if (debug>=2) displayVectorOrMatrix ("softmax probabilities", probabilities, NUM_OUTPUTS, 1);

		for (o=0; o<NUM_OUTPUTS; o++)
            outputErrors[o] = probabilities[o] - y[o]; //the error for each output
		if (debug>=2) displayVectorOrMatrix ("output error", outputErrors, NUM_OUTPUTS, 1);

		for (loss=0, o=0; o<NUM_OUTPUTS; o++)
            loss -= y[o] * log (probabilities[o]); //the loss
		if (debug>=1) printf ("Iteration: %10d, loss: %10.5f\n", t+1, loss); //output the loss

		/// Back propagation --------------------------------------------------
		//Multiply h*e to get the adjustments to deltaWhy
		for (h=0; h<NUM_HIDDEN+1; h++)
            for (int o=0; o<NUM_OUTPUTS; o++)
                deltaWhy[h][o] = hActivationValues[h] * outputErrors[o];
		if (debug>=2) displayVectorOrMatrix ("hidden/output weights gradient", (float*)deltaWhy, NUM_HIDDEN+1, NUM_OUTPUTS);
		//Backward propogate the errors and store in the hActivationValues vector
		memset (hActivationValues, 0, sizeof (hActivationValues)); //set all the weighted sums to zero
		for (h=1; h<NUM_HIDDEN+1; h++)
            for (o=0; o<NUM_OUTPUTS; o++)
                hActivationValues[h] += Why[h][o] * outputErrors[o]; //multiply and sum inputs * weights
		if (debug>=2) displayVectorOrMatrix ("back propagated error values", hActivationValues, NUM_HIDDEN+1, 1);

		for (h=0; h<NUM_HIDDEN; h++)
            zhWeightedSums[h] = hActivationValues[h+1] * (1 - pow (tanh (zhWeightedSums[h]), 2)); //apply activation function gradient
		if (debug>=2) displayVectorOrMatrix ("hidden weighted sums after gradient", zhWeightedSums, NUM_HIDDEN, 1);

		//Multiply x*eh*zh to get the adjustments to deltaWxh, this does not include the bias node
		for (int i=0; i<NUM_INPUTS+1; i++)
            for (int h=0; h<NUM_HIDDEN; h++)
                deltaWxh[i][h] = x[i] * zhWeightedSums[h];
		if (debug>=2) displayVectorOrMatrix ("input/hidden weights gradient", (float*)deltaWxh, NUM_INPUTS+1, NUM_HIDDEN);

		/// Now add in the adjustments ----------------------------------------
		for (int h=0; h<NUM_HIDDEN+1; h++)
            for (int o=0; o<NUM_OUTPUTS; o++)
                Why[h][o] -= learningrate * deltaWhy[h][o];

		for (int i=0; i<NUM_INPUTS+1; i++)
            for (int h=0; h<NUM_HIDDEN; h++)
                Wxh[i][h] -= learningrate * deltaWxh[i][h];
	}

	printf ("------------------------------- Results --------------------------------\n");
	///Output weights and last input/output and predicted output
	displayVectorOrMatrix ("input/hidden weights(Wxh)", (float*)Wxh, NUM_INPUTS+1, NUM_HIDDEN);
	displayVectorOrMatrix ("hidden/output weights(Why)", (float*)Why, NUM_HIDDEN+1, NUM_OUTPUTS);
	displayVectorOrMatrix ("last input", &x[1], NUM_INPUTS, 1);
	displayVectorOrMatrix ("last output", y, NUM_OUTPUTS, 1);
	displayVectorOrMatrix ("predicted output", probabilities, NUM_OUTPUTS, 1);
	return 0;
}
