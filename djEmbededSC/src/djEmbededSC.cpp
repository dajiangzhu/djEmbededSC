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
		int sampleElementNumber, int featureNumber, int startIndex, string strTaskName) {
	std::cout << "readInitialdDictionary begin... " << std::endl;
	double **initialDictioary = InitializeDataM(sampleElementNumber,
			featureNumber);
	stringstream tmpstream;
	tmpstream << "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"<<strTaskName<<"/"
			<< subStartID << "_" << subEndID << "/" << subID << "/OptDicIndex_"
			<< startIndex << "/sub_" << subID << "_OptDicIndex_" << startIndex
			<< "_Round_1_D.txt";
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
		int sampleElementNumber, int startIndex, string strTaskName) {
	double **templateSignal = InitializeDataM(sampleElementNumber, 1);
	stringstream tmpstream;
	tmpstream << "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"<<strTaskName<<"/"
			<< subStartID << "_" << subEndID << "/TemplateSig_" << startIndex
			<< ".txt";
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

double **readErrorM(int subNum, string strTaskName) {
	cout << "Begin to read ErrorDistribution information...";
	double **errorLimitM = InitializeDataM(subNum, 4);
	stringstream tmpstream;
//    tmpstream << "../NIPS2014/optimizedDMatrix/errorLimit.txt";
	tmpstream
			<< "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"<<strTaskName<<"/errorLimit_1_68.txt";
	char errorLimitName[200];
	tmpstream >> errorLimitName;
	cout << "Loading " << errorLimitName << endl;
	FILE *fp;
	fp = fopen(errorLimitName, "rw");
	if (fp == NULL) {
		printf("could not find template signal file %s\n", errorLimitName);
		exit(0);
	}
	for (unsigned int i = 0; i < subNum; i++) {
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
	int epochNumber = 3; // Experienced based
	int featureNumber = 400;
	int sampleElementNumber = 284;
	double lambda = 0.08;
	bool NonNegativeState = false;
	double dStepNum = 10.0;

	//***********************Input*******************************************//
	if (argc == 6) {
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

		//***********************Sparse Learning for each individual*******************************************//
		double** errorLimitM;
		errorLimitM = readErrorM(subNum,strTaskName);
		cout << "#################### Dealing with sub" << subID << endl;
		stringstream signalname;
		signalname
				<< "/ifs/loni/faculty/thompson/four_d/dzhu/data/HCP/TaskFMRI/Whole_b_signals/"<<strTaskName<<"/"
				<< subID << "."<<strTaskName<<".sig.txt";
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
		std::cout << "subNum is " << subNum << std::endl;

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
				nStartIndex, strTaskName);
		double** templateSignal;
		std::cout << "Begin to read template signal." << std::endl;
		templateSignal = readTemplateSignal(strSubStartID, strSubEndID,
				sampleElementNumber, nStartIndex, strTaskName);
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
			Dname
					<< "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"<<strTaskName<<"/"
					<< strSubStartID << "_" << strSubEndID << "/" << subID
					<< "/OptDicIndex_" << nStartIndex << "/sub_" << subID
					<< "_OptDicIndex_" << nStartIndex << "_Round_"
					<< (nStepCont + 1) << "_D.txt";

			Recordname
					<< "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"<<strTaskName<<"/"
					<< strSubStartID << "_" << strSubEndID << "/" << subID
					<< "/config_sub_" << subID << ".txt";
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

				if (totalDecError <= errorLimitM[subID - 1][3]) {
					dpl::saveDictionary(featureNumber, sampleElementNumber, Wd,
							savedDictionaryName);
//					dpl::saveFeature(feature, FeatureFileName, featureNumber,
//							sampleNumber);
					//save feature to the lastRoundFeature
					for (int i = 0; i < sampleNumber; i++)
						for (int j = 0; j < featureNumber; j++)
							lastRoundFeature[i][j] = feature[i][j];
					savedLastRoundFeature = true;

					ofstream out(recordFileName, ios::app);
					out << nStartIndex << " " << (nStepCont + 1) << " "
							<< totalDecError<<endl;
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
		if (savedLastRoundFeature) {
			cout<<"##############   Begin to write feature of the last round!   ##############"<<endl;
			stringstream LastAname;
			LastAname
					<< "/ifs/loni/faculty/thompson/four_d/dzhu/Journal_ESL/results/"<<strTaskName<<"/"
					<< strSubStartID << "_" << strSubEndID << "/" << subID
					<< "/OptDicIndex_" << nStartIndex << "/sub_" << subID
					<< "_OptDicIndex_" << nStartIndex << "_Round_" << nStepCont
					<< "_A.txt";
			char LastFeatureFileName[300];
			LastAname >> LastFeatureFileName;
			dpl::saveFeature(lastRoundFeature, LastFeatureFileName,
					featureNumber, sampleNumber);
			clearDoubleM(sampleNumber, lastRoundFeature);
			cout<<"##############   End of writing feature of the last round!   ##############"<<endl;
		}

		clearDoubleM(sampleElementNumber, javaInitalizedDictioary);
		clearDoubleM(sampleElementNumber, templateSignal);
		clearDoubleM(sampleElementNumber, deltaM);
		dpl::clearSample(sampleNumber, sample);
		std::cout << "Hello World!" << std::endl;
		//    } //for all subjects
		return 0;
	} //if
	else
		cout
				<< "Need paramaters: subStartID subEndID subID(1-68) optDicIndex(0-399) taskName(MOTOR etc.)"
				<< endl;

}
