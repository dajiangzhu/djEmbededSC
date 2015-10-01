/*
 * djEmbededSC_Demo.cpp
 *
 *  Created on: Sep 30, 2015
 *      Author: dzhu
 */
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
		string strESLDir) {
	std::cout << "readInitialdDictionary begin... " << std::endl;
	double **initialDictioary = InitializeDataM(sampleElementNumber,
			featureNumber);
	stringstream tmpstream;
	tmpstream << strESLDir << "/" << subStartID << "_" << subEndID << "/"
			<< subID << "/OptDicIndex_" << startIndex << "/sub_" << subID
			<< "_OptDicIndex_" << startIndex << "_Round_1_D.txt";
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

double **readTemplateSignal(string subStartID, string subEndID,
		int sampleElementNumber, int startIndex, string strESLDir) {
	double **templateSignal = InitializeDataM(sampleElementNumber, 1);
	stringstream tmpstream;
	tmpstream << strESLDir << "/" << subStartID << "_" << subEndID
			<< "/TemplateSig_" << startIndex << ".txt";
	char templateSignalName[500];
	tmpstream >> templateSignalName;
	FILE *fp;
	fp = fopen(templateSignalName, "rw");
	if (fp == NULL) {
		printf("could not find template signal file %s\n", templateSignalName);
		exit(0);
	}
	for (unsigned int i = 0; i < sampleElementNumber; i++) {
		fscanf(fp, "%lf", &templateSignal[i][0]);
	}
	fclose(fp);
	return templateSignal;
}

double **readErrorM(int subNum, string strMutiRunDir) {
	cout << "Begin to read ErrorDistribution information...";
	double **errorLimitM = InitializeDataM(68, 4);
	stringstream tmpstream;
//    tmpstream << "../NIPS2014/optimizedDMatrix/errorLimit.txt";
	tmpstream << strMutiRunDir << "/errorLimit.txt";
	char errorLimitName[200];
	tmpstream >> errorLimitName;
	cout << "Loading " << errorLimitName << endl;
	FILE *fp;
	fp = fopen(errorLimitName, "rw");
	if (fp == NULL) {
		printf("could not find template signal file %s\n", errorLimitName);
		exit(0);
	}
	for (unsigned int i = 0; i < 68; i++) {
		for (int j = 0; j < 4; j++)
			fscanf(fp, "%lf", &errorLimitM[i][j]);
	}
	fclose(fp);
	return errorLimitM;
}

double **getDeltaM(int sampleElementNumber, int startIndex,
		double **initialDictioary, double **templateSignal, double dStepNum) {
	double **deltaM = InitializeDataM(sampleElementNumber, 1);
	for (int i = 0; i < sampleElementNumber; i++)
		deltaM[i][0] = (templateSignal[i][0] - initialDictioary[i][startIndex])
				/ dStepNum;
	return deltaM;
}

bool DictionaryManipulation(double **Wd, int sampleElementNumber,
		int startIndex, double **initialDictioary, double** deltaM,
		int nStepCont) {
	double diff = 0.0;
	for (int i = 0; i < sampleElementNumber; i++)
		diff += deltaM[i][0];
	//copy fixed items
	for (int i = 0; i < sampleElementNumber; i++)
		for (int j = 0; j <= startIndex; j++)
			Wd[i][j] = initialDictioary[i][j];

	if (diff == 0.0)
		return true;
	//adjust current item
	for (int i = 0; i < sampleElementNumber; i++)
		Wd[i][startIndex] += nStepCont * deltaM[i][0];
	return false;
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
	float epochNumber = 3.0; // Experienced based
	int featureNumber = 400;
	int sampleElementNumber = 284;
	double lambda = 0.08;
	bool NonNegativeState = false;
	double dStepNum = 10.0;

	//***********************Input*******************************************//
	if (argc == 13) {
		string strSubStartID = argv[1]; //0-399
		int subStartID = atoi(strSubStartID.c_str());
		string strSubEndID = argv[2]; //0-399
		int subEndID = atoi(strSubEndID.c_str());
		string strSubID = argv[3]; //0-399
		int subID = atoi(strSubID.c_str());
		string strStartIndex = argv[4]; //0-399
		int nStartIndex = atoi(strStartIndex.c_str());
		int subNum = subEndID - subStartID + 1;
		string strOriDataDir = argv[5]; //0-399
		string strMutiRunDir = argv[6]; //0-399
		string strESLDir = argv[7];
		string strFeatureNumber = argv[8];
		featureNumber = atoi(strFeatureNumber.c_str());
		string strSampleElementNumber = argv[9];
		sampleElementNumber = atoi(strSampleElementNumber.c_str());
		string strEpochNumber = argv[10];
		epochNumber = atof(strEpochNumber.c_str());
		string strLambda = argv[11];
		lambda = atof(strLambda.c_str());
		string strStepNum = argv[12];
		dStepNum = atof(strStepNum.c_str());

		//***********************Sparse Learning for each individual*******************************************//
		double** errorLimitM;
		errorLimitM = readErrorM(subNum, strMutiRunDir);
		cout << "#################### Dealing with sub" << subID << endl;
		stringstream signalname;
		signalname << strOriDataDir << "/sub." << subID << ".txt";
		char SampleFileName[100];
		signalname >> SampleFileName;

		double **sample;
		int sampleNumber = dpl::getSampleNumber(SampleFileName);
		int iterationNumber = sampleNumber * epochNumber;

		std::cout << "strOriDataDir is " << strOriDataDir << std::endl;
		std::cout << "strMutiRunDir is " << strMutiRunDir << std::endl;
		std::cout << "strESLDir is " << strESLDir << std::endl;
		std::cout << "Number of samples is " << sampleNumber << std::endl;
		std::cout << "Number of samples' element is " << sampleElementNumber
				<< std::endl;
		std::cout << "Number of features is " << featureNumber << std::endl;
		std::cout << "Number of sampleElementNumber is " << sampleElementNumber
				<< std::endl;
		std::cout << "Number of epochNumber is " << epochNumber << std::endl;
		std::cout << "lambda is " << lambda << std::endl;
		std::cout << "Number of Iterations is " << iterationNumber << std::endl;
		std::cout << "subStartID is " << subStartID << std::endl;
		std::cout << "subEndID is " << subEndID << std::endl;
		std::cout << "subID is " << subID << std::endl;
		std::cout << "nStartIndex is " << nStartIndex << std::endl;
		std::cout << "subNum is " << subNum << std::endl;
		std::cout << "dStepNum is " << dStepNum << std::endl;

		std::cout << "Begin to read sample." << std::endl;
		sample = dpl::ReadSample(SampleFileName, sampleNumber,
				sampleElementNumber);
		std::cout << "Begin to normalize sample." << std::endl;
		dpl::SampleNormalization(sample, sampleNumber, sampleElementNumber);

		int nStepCont = 1;
		double** javaInitalizedDictioary;
		std::cout << "Begin to read initial dictionary." << std::endl;
		javaInitalizedDictioary = readInitialdDictionary(strSubStartID,
				strSubEndID, subID, sampleElementNumber, featureNumber,
				nStartIndex, strESLDir);
		double** templateSignal;
		std::cout << "Begin to read template signal." << std::endl;
		templateSignal = readTemplateSignal(strSubStartID, strSubEndID,
				sampleElementNumber, nStartIndex, strESLDir);
		double** deltaM;
		deltaM = getDeltaM(sampleElementNumber, nStartIndex,
				javaInitalizedDictioary, templateSignal, dStepNum);
		double **lastRoundFeature = InitializeDataM(sampleNumber,
				featureNumber);
		bool savedLastRoundFeature = false;

		bool reachTemplateSig = false;
		double totalDecError = 0.0;
		do {
			cout << "-----------Round: " << nStepCont << endl;
			double **Wd;
			double **feature;
			stringstream Dname;
			stringstream Recordname;
			Dname << strESLDir << "/" << strSubStartID << "_" << strSubEndID
					<< "/" << subID << "/OptDicIndex_" << nStartIndex << "/sub_"
					<< subID << "_OptDicIndex_" << nStartIndex << "_Round_"
					<< (nStepCont + 1) << "_D.txt";

			Recordname << strESLDir << "/" << strSubStartID << "_"
					<< strSubEndID << "/" << subID << "/config_sub_" << subID
					<< ".txt";
			char savedDictionaryName[300];
			char recordFileName[300];
			Dname >> savedDictionaryName;
			Recordname >> recordFileName;

			//Initialize random dictionary
			Wd = dpl::GenerateRandomPatchDictionary(featureNumber,
					sampleElementNumber, sampleNumber, sample);
			dpl::DictionaryNormalization(featureNumber, sampleElementNumber,
					Wd);

			//Set the fixed template signals and update the current signals (at startIndex))
			reachTemplateSig = DictionaryManipulation(Wd, sampleElementNumber,
					nStartIndex, javaInitalizedDictioary, deltaM, nStepCont);
			cout << "reachTemplateSig = " << reachTemplateSig << endl;
			if (!reachTemplateSig) {
				//Begin Sparse Learning
				feature = dpl::FeatureInitialization(featureNumber,
						sampleNumber);
				std::cout << "Begin to train " << std::endl;
				totalDecError = dpl::trainDecoder(Wd, feature, sample, lambda,
						layers, featureNumber, sampleNumber,
						sampleElementNumber, iterationNumber, NonNegativeState,
						nStartIndex);
				std::cout << "Finish training " << std::endl;
				std::cout << "errorLimitM is: " << errorLimitM[subID - 1][3]
						<< std::endl;

				if (totalDecError <= errorLimitM[subID - 1][3]
						|| nStepCont == 1) {
					std::cout << "Save DMatrix... " << std::endl;
					dpl::saveDictionary(featureNumber, sampleElementNumber, Wd,
							savedDictionaryName);
//					dpl::saveFeature(feature, FeatureFileName, featureNumber,
//							sampleNumber);
					//save feature to the lastRoundFeature
//					for (int i = 0; i < sampleNumber; i++)
//						for (int j = 0; j < featureNumber; j++)
//							lastRoundFeature[i][j] = feature[i][j];
//					savedLastRoundFeature = true;

					ofstream out(recordFileName, ios::app);
					out << nStartIndex << " " << (nStepCont + 1) << " "
							<< totalDecError << endl;
					out.close();
				} else
					cout << "!!!Will quit!  totalDecError:" << totalDecError
							<< "  errorLimit:" << errorLimitM[subID - 1][3]
							<< endl;

				dpl::clearFeature(sampleNumber, feature);
			}
			dpl::clearDictionary(sampleElementNumber, Wd);
			nStepCont++;
		} while (!reachTemplateSig && nStepCont <= dStepNum
				&& totalDecError <= errorLimitM[subID - 1][3]);
		//write the last round feature
//		if (savedLastRoundFeature) {
//			cout<<"##############   Begin to write feature of the last round!   ##############"<<endl;
//			stringstream LastAname;
//			LastAname
//					<< "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"<<strTaskName<<"/"
//					<< strSubStartID << "_" << strSubEndID << "/" << subID
//					<< "/OptDicIndex_" << nStartIndex << "/sub_" << subID
//					<< "_OptDicIndex_" << nStartIndex << "_Round_" << nStepCont
//					<< "_A.txt";
//			char LastFeatureFileName[300];
//			LastAname >> LastFeatureFileName;
//			dpl::saveFeature(lastRoundFeature, LastFeatureFileName,
//					featureNumber, sampleNumber);
//			clearDoubleM(sampleNumber, lastRoundFeature);
//			cout<<"##############   End of writing feature of the last round!   ##############"<<endl;
//		}

		clearDoubleM(sampleElementNumber, javaInitalizedDictioary);
		clearDoubleM(sampleElementNumber, templateSignal);
		clearDoubleM(sampleElementNumber, deltaM);
		dpl::clearSample(sampleNumber, sample);
		//
		stringstream flagFile;
		flagFile << strESLDir << "/" << strSubStartID << "_" << strSubEndID
				<< "/" << subID << "/OptDicIndex_" << nStartIndex << "/sub_"
				<< subID << "_OptDicIndex_" << nStartIndex << "_flag.txt";
		char recordFileName[300];
		flagFile >> recordFileName;
		ofstream outFlag(recordFileName, ios::out);
		outFlag <<"nStepCont:"<<nStepCont<< endl;
		outFlag.close();

		std::cout << "Hello World!" << std::endl;
		//    } //for all subjects
		return 0;
	} //if
	else
		cout
				<< "Need paramaters: subStartID subEndID subID(1-68) optDicIndex(0-399) strOriDataDir strMutiRunDir strESLDir strFeatureNumber(400) strSampleElementNumber(284) strEpochNumber lambda(0.08) strStepNum(10)"
				<< endl;

}

