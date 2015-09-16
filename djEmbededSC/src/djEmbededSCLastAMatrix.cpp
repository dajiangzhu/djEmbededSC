/*
 Sparse Coordinate Coding  version 1.0.2
 */
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <iterator>
#include <vector>
#include <cmath>
#include <stdio.h>
#include <time.h>
#include <ctime>
#include <iomanip>
#include <string>
#include <omp.h>
#include "DictionaryGeneration.h"
#include "SampleNormalization.h"
#include "LR.h"
#include "SCC.h"
using namespace std;

//***************Introduced by DJ.ZHU****Begin

double **InitializeDataM(int rowNum, int columnNum) {

	double **dataM = (double**) malloc(rowNum * sizeof(double*));
	for (unsigned int i = 0; i < rowNum; i++)
		dataM[i] = (double*) malloc(columnNum * sizeof(double));
	return dataM;
}

double **readInitialdDictionary(string subStartID, string subEndID, int subID,
		int sampleElementNumber, int featureNumber, int startIndex,
		string strTaskName, string initialRoundIndex) {
	std::cout << "readInitialdDictionary begin... " << std::endl;
	double **initialDictioary = InitializeDataM(sampleElementNumber,
			featureNumber);
	stringstream tmpstream;
	tmpstream << "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"
			<< strTaskName << "/" << subStartID << "_" << subEndID << "/"
			<< subID << "/OptDicIndex_" << startIndex << "/sub_" << subID
			<< "_OptDicIndex_" << startIndex << "_Round_" << initialRoundIndex
			<< "_D.txt";
	char initialDictionaryName[300];
	tmpstream >> initialDictionaryName;
	std::cout << "initialDictionaryName " << initialDictionaryName << std::endl;
	FILE *fp;
	fp = fopen(initialDictionaryName, "rw");
	if (fp == NULL) {
		printf("could not find template signal file %s\n",
				initialDictionaryName);
		exit(0);
	}
	for (unsigned int i = 0; i < sampleElementNumber; i++) {
		for (unsigned int j = 0; j < featureNumber; j++)
			fscanf(fp, "%lf", &initialDictioary[i][j]);
	}
	fclose(fp);
	return initialDictioary;
}

void DictionaryManipulation(double **Wd, int sampleElementNumber,
		int startIndex, double **initialDictioary) {
	//copy fixed items
	for (int i = 0; i < sampleElementNumber; i++)
		for (int j = 0; j <= startIndex; j++)
			Wd[i][j] = initialDictioary[i][j];
}

void clearDoubleM(int rowNum, double **dataM) {
	for (unsigned int i = 0; i < rowNum; i++) {
		free(dataM[i]);
	}
	free(dataM);
}

//***************Introduced by DJ.ZHU********End

int main(int argc, char* argv[]) {

	//***********************General Defination*******************************************//
	int layers = 3;
	int epochNumber = 3; // Experienced based
	int featureNumber = 400;
	int sampleElementNumber = 284;
	double lambda = 0.08;
	bool NonNegativeState = false;

	//***********************Input*******************************************//
	if (argc == 7) {
		string strSubStartID = argv[1]; //0-399
		int subStartID = atoi(strSubStartID.c_str());
		string strSubEndID = argv[2]; //0-399
		int subEndID = atoi(strSubEndID.c_str());
		string strSubID = argv[3]; //0-399
		int subID = atoi(strSubID.c_str());
		string strStartIndex = argv[4]; //0-399
		int nStartIndex = atoi(strStartIndex.c_str());
		int subNum = subEndID - subStartID + 1;
		string strTaskName = argv[5]; //0-399
		string initialRoundIndex = argv[6];

		//***********************Sparse Learning for each individual*******************************************//
		cout << "#################### Dealing with sub" << subID << endl;
		stringstream signalname;
		signalname
				<< "/ifs/loni/faculty/thompson/four_d/dzhu/data/HCP/TaskFMRI/Whole_b_signals/"
				<< strTaskName << "/" << subID << "." << strTaskName
				<< ".sig.txt";
		char SampleFileName[100];
		signalname >> SampleFileName;

		double **sample;
		int sampleNumber = dpl::getSampleNumber(SampleFileName);
		int iterationNumber = sampleNumber * epochNumber;

		std::cout << "Number of samples is " << sampleNumber << std::endl;
		std::cout << "Number of samples' element is " << sampleElementNumber
				<< std::endl;
		std::cout << "Number of features is " << featureNumber << std::endl;
		std::cout << "Number of Iterations is " << iterationNumber << std::endl;
		std::cout << "lambda is " << lambda << std::endl;
		std::cout << "subStartID is " << subStartID << std::endl;
		std::cout << "subEndID is " << subEndID << std::endl;
		std::cout << "subID is " << subID << std::endl;
		std::cout << "nStartIndex is " << nStartIndex << std::endl;
		std::cout << "initialRoundIndex is " << initialRoundIndex << std::endl;
		std::cout << "subNum is " << subNum << std::endl;

		std::cout << "Begin to read sample." << std::endl;
		sample = dpl::ReadSample(SampleFileName, sampleNumber,
				sampleElementNumber);
		std::cout << "Begin to normalize sample." << std::endl;
		dpl::SampleNormalization(sample, sampleNumber, sampleElementNumber);

		double** javaInitalizedDictioary;
		std::cout << "Begin to read initial dictionary." << std::endl;
		javaInitalizedDictioary = readInitialdDictionary(strSubStartID,
				strSubEndID, subID, sampleElementNumber, featureNumber,
				nStartIndex, strTaskName, initialRoundIndex);

		double totalDecError = 0.0;
		double **Wd;
		double **feature;
		stringstream Dname;
		stringstream Recordname;
		Dname << "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"
				<< strTaskName << "/" << strSubStartID << "_" << strSubEndID
				<< "/" << subID << "/OptDicIndex_" << nStartIndex << "/sub_"
				<< subID << "_OptDicIndex_" << nStartIndex
				<< "_Round_final_D.txt";
		stringstream LastAname;
		LastAname
				<< "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"
				<< strTaskName << "/" << strSubStartID << "_" << strSubEndID
				<< "/" << subID << "/OptDicIndex_" << nStartIndex << "/sub_"
				<< subID << "_OptDicIndex_" << nStartIndex
				<< "_Round_final_A.txt";
		char LastFeatureFileName[300];
		LastAname >> LastFeatureFileName;

		char savedDictionaryName[300];
		Dname >> savedDictionaryName;

		//Initialize random dictionary
		Wd = dpl::GenerateRandomPatchDictionary(featureNumber,
				sampleElementNumber, sampleNumber, sample);
		dpl::DictionaryNormalization(featureNumber, sampleElementNumber, Wd);

		//Set the fixed template signals and update the current signals (at startIndex))
		DictionaryManipulation(Wd, sampleElementNumber,
				nStartIndex, javaInitalizedDictioary);
		//Begin Sparse Learning
		feature = dpl::FeatureInitialization(featureNumber, sampleNumber);
		std::cout << "Begin to train " << std::endl;
		totalDecError = dpl::trainDecoder(Wd, feature, sample, lambda, layers,
				featureNumber, sampleNumber, sampleElementNumber,
				iterationNumber, NonNegativeState, nStartIndex);
		std::cout << "Finish training with Error: " << totalDecError
				<< std::endl;

		cout
				<< "##############   Begin to write Dictionary ...   ##############"
				<< endl;
		dpl::saveDictionary(featureNumber, sampleElementNumber, Wd,
				savedDictionaryName);
		cout << "##############   Begin to write Feature ...   ##############"
				<< endl;
		dpl::saveFeature(feature, LastFeatureFileName, featureNumber,
				sampleNumber);

		dpl::clearFeature(sampleNumber, feature);
		dpl::clearDictionary(sampleElementNumber, Wd);

		clearDoubleM(sampleElementNumber, javaInitalizedDictioary);
		dpl::clearSample(sampleNumber, sample);
		std::cout << "Hello World!" << std::endl;
		//    } //for all subjects
		return 0;
	} //if
	else
		cout
				<< "Need paramaters: subStartID subEndID subID(1-68) optDicIndex(0-399) taskName(MOTOR etc.) initialRoundIndex"
				<< endl;

}
