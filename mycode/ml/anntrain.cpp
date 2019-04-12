#include <opencv2/core/core.hpp>
#include <opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include "ImageFeatrueExtract.h"

#include <iostream>
#include <string>
#include <fstream>
#include <atltime.h>
#include <windows.h>
#include <stdlib.h>
#include <io.h>
#include <direct.h>
#include <stdio.h>

using namespace std;
using namespace cv;

#define WIDTH	36
#define HEIGHT	60
#define LGRIDX	4
#define LGRIDY	4
#define HOGBINNUM 9
#define HOGGRIDX 4
#define HOGGRIDY 4

const char PNName[] = "F:\\MyFiles\\MyCode\\AdaAndANN\\AdaAndANN\\BadFist";  
const char PPName[] = "F:\\trainImageData\\FineSamples\\FinePalm";
const char PPName_1[] = "F:\\MyFiles\\MyCode\\AdaAndANN\\AdaAndANN\\FineFist_1";

const char NPName[] = "E:\\trainImageData\\FistData\\NPFist";
const char NNName[] = "F:\\trainImageData\\FineSamples\\FineNeg";

const string palm = "F:\\samples\\palm";
const string thumbUp = "F:\\samples\\thumbUp";
const string fist = "F:\\samples\\fist";
const string neg = "F:\\samples\\neg";
const string face = "F:\\samples\\face";
const string sideFace = "F:\\samples\\sideFace";
const string halfFace = "F:\\samples\\halfFace";
const string drape = "F:\\samples\\drape";
const string sign = "F:\\samples\\sign";
const string stripe = "F:\\samples\\stripe";
const string inclineFace = "F:\\samples\\inclineFace";

char fileName[200];
char timeName[30];
const string PNFileName = "F:\\MyFiles\\MyCode\\AdaAndANN\\AdaAndANN\\BadFist";
const string PPFileName = "F:\\trainImageData\\FineSamples\\FinePalm";
const string PPFileName_1 = "F:\\MyFiles\\MyCode\\AdaAndANN\\AdaAndANN\\FineFist_1";

const string NPFileName = "E:\\trainImageData\\FistData\\NPFist";
const string NNFileName = "F:\\trainImageData\\FineSamples\\FineNeg";


string getAnyLine(string filename, int index);
string getFileName(const char* , int);
//#define series_connection  //for test

//for train
int main_train_test()
{
	try{

		//txt文件的每一行是一张图像的全路径
		string strPosFileListForTrain = "E:\\pythonProject\\PosFistData0711.txt";		//用于训练的正样本描述文件
		string strNegFileListForTrain = "E:\\pythonProject\\NegForFistData0711.txt";			//用于训练的负样本描述文件
		string strPosFileListForTest = "E:\\pythonProject\\PosForFistTest0711.txt";			//用于测试的正样本描述文件
		string strNegFileListForTest = "‪E:\\pythonProject\\NegForFistTest0711.txt";					//用于测试的负样本描述文件

		//训练阶段
		cout<<"data prepare stage"<<endl;
		cout<<"train samples feature extract"<<endl;
		Mat matPosForTrain, matNegForTrain, matAllForTrain;
		int ip = 0;
		int in = 0;
		cout<<"pos train data prepare"<<endl;
		// 	ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strPosFileListForTrain, matPosForTrain, ip, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );

		//ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strPosFileListForTrain, matPosForTrain, ip, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );
		ImageFeatureLib::Read_List_Get_TPLBP_Hog_Grid(strPosFileListForTrain, matPosForTrain, ip, false, true, 2, 36, 36, HOGBINNUM, HOGGRIDX, HOGGRIDY, LGRIDX, LGRIDY);
		int iPosCount = matPosForTrain.rows;
		matAllForTrain.push_back(matPosForTrain);
		matPosForTrain.release();
		cout<<"pos data done"<<endl;

		cout<<"neg train data prepare"<<endl;
		//ImageFeatureLib::Read_List_Get_HOG_Grid( strNegFileListForTrain, matNegForTrain, in, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );
		//ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strNegFileListForTrain, matNegForTrain, in, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );
		ImageFeatureLib::Read_List_Get_TPLBP_Hog_Grid(strNegFileListForTrain, matNegForTrain, in, false, true, 2, 36, 36, HOGBINNUM, HOGGRIDX, HOGGRIDY, LGRIDX, LGRIDY);
		int iNegCount = matNegForTrain.rows;
		matAllForTrain.push_back(matNegForTrain);
		matNegForTrain.release();
		cout<<"neg data done"<<endl;

		int iAllCount = matAllForTrain.rows;
		cout<<"pos count:"<<iPosCount<<endl;
		cout<<"neg count:"<<iNegCount<<endl;
		cout<<"all count:"<<iAllCount<<endl;

		cout<<"label prepare"<<endl;
		Mat matLabels( iAllCount, 2, CV_32FC1, Scalar(0) );
		for ( int iRow=0; iRow<iPosCount; iRow++ )
		{
			matLabels.ptr<float>(iRow)[0]=1;
			matLabels.ptr<float>(iRow)[1]=0;
		}
		for ( int iRow=iPosCount; iRow<iAllCount; iRow++ )
		{
			matLabels.ptr<float>(iRow)[0]=0;
			matLabels.ptr<float>(iRow)[1]=1;
		}

		cout<<"set ann parameters..."<<endl;
		//CvANN_MLP_TrainParams params;
		//params.train_method = CvANN_MLP_TrainParams::BACKPROP;
		//params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 200, 0.0001 );
		//params.bp_dw_scale = 0.0001;
		//params.bp_moment_scale = 0.1;

		//CvANN_MLP ann;
		Ptr<ml::ANN_MLP> ann = ml::ANN_MLP::create();
		int iInputLayer = matAllForTrain.cols;
		Mat matLayerSize = (Mat_<int>(1,3) << iInputLayer,64, 2);//layer size  网络层数
		cout<<"layerSize:"<<matLayerSize<<endl;
		//ann.create( matLayerSize, CvANN_MLP::SIGMOID_SYM );
		ann->setLayerSizes(matLayerSize);
		ann->setActivationFunction(ml::ANN_MLP::SIGMOID_SYM, 1, 1);
		//ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001,0.1);
		ann->setTrainMethod(ml::ANN_MLP::BACKPROP, 0.001);

		double dfTrainStartTime = (double)cv::getTickCount();
		ann->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER+TermCriteria::EPS,300,FLT_EPSILON));
		Ptr<ml::TrainData> tData = ml::TrainData::create(matAllForTrain,ml::ROW_SAMPLE,matLabels);
		cout << "Begin training: ..." << endl;
		ann->train(tData);
		//ann.train( matAllForTrain, matLabels, Mat(), Mat(), params );//train model
		double dfTrainEndTime = ( (double)cv::getTickCount() - dfTrainStartTime ) * 1000 / cv::getTickFrequency();
		cout<<"train time:"<<dfTrainEndTime<<"ms"<<endl;
		ann->save("FistModel_TPLBP_Hog_3layer0713_99.xml");//save model
		cout<<"Save mode success!"<<endl;


		//测试阶段
		cout<<"test stage"<<endl;
		int iPosPosTrain=0, iPosNegTrain=0, iNegPosTrain=0, iNegNegTrain=0;

		cout<<"test train samples"<<endl;
		for (int iRow=0; iRow<iPosCount; iRow++)
		{
			Mat matTmp = matAllForTrain.row(iRow);
			Mat matResult;
			ann->predict( matTmp, matResult );
			//float *pfRow = matResult.ptr<float>(0);
			float response = matResult.ptr<float>(0)[0];
			//if ( pfRow[0]>pfRow[1] )
			if (response > 0.75)
				iPosPosTrain++;
			else
				iPosNegTrain++;
		}
		for (int iRow=iPosCount; iRow<iAllCount; iRow++)
		{
			Mat matTmp = matAllForTrain.row(iRow);
			Mat matResult;
			ann->predict( matTmp, matResult );
			//float *pfRow = matResult.ptr<float>(0);
			float response = matResult.ptr<float>(0)[0];
			//if ( pfRow[0]>pfRow[1] )
			if (response > 0.75)
				iNegPosTrain++;
			else
				iNegNegTrain++;
		}
		matAllForTrain.release();
		double dfPosPrecisionTrain = (double)iPosPosTrain / iPosCount;
		double dfNegPrecisionTrain = (double)iNegNegTrain / iNegCount;
		double dfAllPrecisionTrain = (double)(iPosPosTrain + iNegNegTrain) / (iPosCount + iNegCount);
		cout<<endl;
		//cout<<"train:"<<iPosPosTrain<<" "<<iPosNegTrain<<" "<<iNegPosTrain<<" "<<iNegNegTrain<<endl;
		cout<<"train:"<<"iPosPosTrain: "<<iPosPosTrain<<" "<<"iPosNegTrain: "<<iPosNegTrain<<" "<<"iNegPosTrain: "<<iNegPosTrain<<" "<<"iNegNegTrain: "<<iNegNegTrain<<endl;
		cout<<"train dfPosPrecision:"<<dfPosPrecisionTrain<<endl;
		cout<<"train dfNegPrecision:"<<dfNegPrecisionTrain<<endl;
		cout<<"train dfAllPrecision:"<<dfAllPrecisionTrain<<endl;

		cout<<"test test samples"<<endl;
		cout<<"test samples feature extract"<<endl;
		double startTime = (double)cv::getTickCount();
		Mat matPosForTest, matNegForTest;
		int itp = 0;
		int itn = 0;
		//ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strPosFileListForTest, matPosForTest, itp, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );
		//ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strNegFileListForTest, matNegForTest, itn, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );

		ImageFeatureLib::Read_List_Get_TPLBP_Hog_Grid(strPosFileListForTest, matPosForTest, itp, false, true, 2, 36, 36, HOGBINNUM, HOGGRIDX, HOGGRIDY, LGRIDX, LGRIDY);
		ImageFeatureLib::Read_List_Get_TPLBP_Hog_Grid(strNegFileListForTest, matNegForTest, itn, false, true, 2, 36, 36, HOGBINNUM, HOGGRIDX, HOGGRIDY, LGRIDX, LGRIDY);
		int iPosCountForTest = matPosForTest.rows;
		int iNegCountForTest = matNegForTest.rows;
		cout<<"feature extract done"<<endl;

		cout<<"predict result"<<endl;
		int iPosPosTest=0, iPosNegTest=0, iNegPosTest=0, iNegNegTest=0;
		for (int iRow=0; iRow<iPosCountForTest; iRow++)
		{
			Mat matTmp=matPosForTest.row(iRow);
			Mat matResult;
			ann->predict( matTmp, matResult );
			//float *pfRow = matResult.ptr<float>(0);
			//if ( pfRow[0]>pfRow[1] )
			float response = matResult.ptr<float>(0)[0];
			if( response > 0.75)
				iPosPosTest++;
			else 
				iPosNegTest++;
		}
		for (int iRow=0; iRow<iNegCountForTest; iRow++)
		{
			Mat matTmp = matNegForTest.row(iRow);
			Mat matResult;
			ann->predict( matTmp, matResult );
			//float *pfRow = matResult.ptr<float>(0);
			//if ( pfRow[0]>pfRow[1])
			float response = matResult.ptr<float>(0)[0];
			if (response > 0.75)
				iNegPosTest++;
			else
				iNegNegTest++;
		}
		double endTime = ( (double)cv::getTickCount() - startTime ) * 1000 / cv::getTickFrequency();
		double meanTime = endTime / ( iPosCountForTest + iNegCountForTest );
		double dfPosPrecisionTest = (double)iPosPosTest / iPosCountForTest;
		double dfNegPrecisionTest = (double)iNegNegTest / iNegCountForTest;
		double dfAllPrecisionTest = (double)( iPosPosTest + iNegNegTest ) / ( iPosCountForTest + iNegCountForTest );

		cout<<endl;
		cout<<"test mean time:"<<meanTime<<"ms"<<endl;
		cout<<"test:"<<"iPosPosTest: "<<iPosPosTest<<" "<<"iPosNegTest: "<<iPosNegTest<<" "<<"iNegPosTest: "<<iNegPosTest<<" "<<"iNegNegTest: "<<iNegNegTest<<endl;
		cout<<"test dfPosPrecision:"<<dfPosPrecisionTest<<endl;
		cout<<"test dfNegPrecision:"<<dfNegPrecisionTest<<endl;
		cout<<"test dfAllPrecision:"<<dfAllPrecisionTest<<endl;

		system("pause");
	}
	catch(Exception &e)
	{
		cout<<e.what()<<endl;
		system("pause");
	}
	return 0;
}


//for train
// int main()
// {
// 	//txt文件的每一行是一张图像的全路径
// 	string strPosFileListForTrain = "E:/pythonProject/forThumbup/thumbupTrain25W.txt";		//用于训练的正样本描述文件
// 	string strNegFileListForTrain = "E:/pythonProject/forNeg/negTrain25W.txt";				//用于训练的负样本描述文件
// 	string strPosFileListForTest = "E:/pythonProject/forThumbup/thumbupTest.txt";			//用于测试的正样本描述文件
// 	string strNegFileListForTest = "E:/pythonProject/forNeg/negTest.txt";					//用于测试的负样本描述文件
// 
// 	//训练阶段
// 	cout<<"data prepare stage"<<endl;
// 	cout<<"train samples feature extract"<<endl;
// 	Mat matPosForTrain, matNegForTrain, matAllForTrain;
// 	int ip = 0;
// 	int in = 0;
// 	cout<<"pos train data prepare"<<endl;
// 	ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strPosFileListForTrain, matPosForTrain, ip, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );
// 	int iPosCount = matPosForTrain.rows;
// 	matAllForTrain.push_back(matPosForTrain);
// 	matPosForTrain.release();
// 	cout<<"pos data done"<<endl;
// 
// 	cout<<"neg train data prepare"<<endl;
// 	ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strNegFileListForTrain, matNegForTrain, in, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );
// 	int iNegCount = matNegForTrain.rows;
// 	matAllForTrain.push_back(matNegForTrain);
// 	matNegForTrain.release();
// 	cout<<"neg data done"<<endl;
// 
// 	int iAllCount = matAllForTrain.rows;
// 	cout<<"pos count:"<<iPosCount<<endl;
// 	cout<<"neg count:"<<iNegCount<<endl;
// 	cout<<"all count:"<<iAllCount<<endl;
// 
// 	cout<<"label prepare"<<endl;
// 	Mat matLabels( iAllCount, 2, CV_32FC1, Scalar(0) );
// 	for ( int iRow=0; iRow<iPosCount; iRow++ )
// 	{
// 		matLabels.ptr<float>(iRow)[0]=1;
// 		matLabels.ptr<float>(iRow)[1]=0;
// 	}
// 	for ( int iRow=iPosCount; iRow<iAllCount; iRow++ )
// 	{
// 		matLabels.ptr<float>(iRow)[0]=0;
// 		matLabels.ptr<float>(iRow)[1]=1;
// 	}
// 	
// 	cout<<"set ann parameters..."<<endl;
// 	CvANN_MLP_TrainParams params;
// 	params.train_method = CvANN_MLP_TrainParams::BACKPROP;
// 	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 200, 0.0001 );
// 	params.bp_dw_scale = 0.0001;
// 	params.bp_moment_scale = 0.1;
// 
// 	CvANN_MLP ann;
// 	int iInputLayer = matAllForTrain.cols;
// 	Mat matLayerSize = (Mat_<int>(1,3) << iInputLayer, 128, 2);//layer size
// 	cout<<"layerSize:"<<matLayerSize<<endl;
// 	ann.create( matLayerSize, CvANN_MLP::SIGMOID_SYM );
// 	double dfTrainStartTime = (double)cv::getTickCount();
// 	ann.train( matAllForTrain, matLabels, Mat(), Mat(), params );//train model
// 	double dfTrainEndTime = ( (double)cv::getTickCount() - dfTrainStartTime ) * 1000 / cv::getTickFrequency();
// 	cout<<"train time:"<<dfTrainEndTime<<"ms"<<endl;
// 	ann.save("model.xml");//save model
// 
// 
// 	//测试阶段
// 	cout<<"test stage"<<endl;
// 	int iPosPosTrain=0, iPosNegTrain=0, iNegPosTrain=0, iNegNegTrain=0;
// 
// 	cout<<"test train samples"<<endl;
// 	for (int iRow=0; iRow<iPosCount; iRow++)
// 	{
// 		Mat matTmp = matAllForTrain.row(iRow);
// 		Mat matResult;
// 		ann.predict( matTmp, matResult );
// 		float *pfRow = matResult.ptr<float>(0);
// 		if ( pfRow[0]>pfRow[1] )
// 			iPosPosTrain++;
// 		else
// 			iPosNegTrain++;
// 	}
// 	for (int iRow=iPosCount; iRow<iAllCount; iRow++)
// 	{
// 		Mat matTmp = matAllForTrain.row(iRow);
// 		Mat matResult;
// 		ann.predict( matTmp, matResult );
// 		float *pfRow = matResult.ptr<float>(0);
// 		if ( pfRow[0]>pfRow[1] )
// 			iNegPosTrain++;
// 		else
// 			iNegNegTrain++;
// 	}
// 	matAllForTrain.release();
// 	double dfPosPrecisionTrain = (double)iPosPosTrain / iPosCount;
// 	double dfNegPrecisionTrain = (double)iNegNegTrain / iNegCount;
// 	double dfAllPrecisionTrain = (double)(iPosPosTrain + iNegNegTrain) / (iPosCount + iNegCount);
// 	cout<<endl;
// 	cout<<"train:"<<iPosPosTrain<<" "<<iPosNegTrain<<" "<<iNegPosTrain<<" "<<iNegNegTrain<<endl;
// 	cout<<"train dfPosPrecision:"<<dfPosPrecisionTrain<<endl;
// 	cout<<"train dfNegPrecision:"<<dfNegPrecisionTrain<<endl;
// 	cout<<"train dfAllPrecision:"<<dfAllPrecisionTrain<<endl;
// 
// 	cout<<"test test samples"<<endl;
// 	cout<<"test samples feature extract"<<endl;
// 	double startTime = (double)cv::getTickCount();
// 	Mat matPosForTest, matNegForTest;
// 	int itp = 0;
// 	int itn = 0;
// 	ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strPosFileListForTest, matPosForTest, itp, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );
// 	ImageFeatureLib::Read_List_Get_TPLBP_Uniform_Grid( strNegFileListForTest, matNegForTest, itn, true, WIDTH, HEIGHT, LGRIDX, LGRIDY );
// 	int iPosCountForTest = matPosForTest.rows;
// 	int iNegCountForTest = matNegForTest.rows;
// 	cout<<"feature extract done"<<endl;
// 
// 	cout<<"predict result"<<endl;
// 	int iPosPosTest=0, iPosNegTest=0, iNegPosTest=0, iNegNegTest=0;
// 	for (int iRow=0; iRow<iPosCountForTest; iRow++)
// 	{
// 		Mat matTmp=matPosForTest.row(iRow);
// 		Mat matResult;
// 		ann.predict( matTmp, matResult );
// 		float *pfRow = matResult.ptr<float>(0);
// 		if ( pfRow[0]>pfRow[1] )
// 			iPosPosTest++;
// 		else
// 			iPosNegTest++;
// 	}
// 	for (int iRow=0; iRow<iNegCountForTest; iRow++)
// 	{
// 		Mat matTmp = matNegForTest.row(iRow);
// 		Mat matResult;
// 		ann.predict( matTmp, matResult );
// 		float *pfRow = matResult.ptr<float>(0);
// 		if ( pfRow[0]>pfRow[1])
// 			iNegPosTest++;
// 		else
// 			iNegNegTest++;
// 	}
// 	double endTime = ( (double)cv::getTickCount() - startTime ) * 1000 / cv::getTickFrequency();
// 	double meanTime = endTime / ( iPosCountForTest + iNegCountForTest );
// 	double dfPosPrecisionTest = (double)iPosPosTest / iPosCountForTest;
// 	double dfNegPrecisionTest = (double)iNegNegTest / iNegCountForTest;
// 	double dfAllPrecisionTest = (double)( iPosPosTest + iNegNegTest ) / ( iPosCountForTest + iNegCountForTest );
// 
// 	cout<<endl;
// 	cout<<"test mean time:"<<meanTime<<"ms"<<endl;
// 	cout<<"test:"<<iPosPosTest<<" "<<iPosNegTest<<" "<<iNegPosTest<<" "<<iNegNegTest<<endl;
// 	cout<<"test dfPosPrecision:"<<dfPosPrecisionTest<<endl;
// 	cout<<"test dfNegPrecision:"<<dfNegPrecisionTest<<endl;
// 	cout<<"test dfAllPrecision:"<<dfAllPrecisionTest<<endl;
// 
// 	system("pause");
// 	return 0;
// }

//for just test
int main()
{
	if (access(PNFileName.c_str(), 0) == -1)
		mkdir(PNFileName.c_str());
	//if (access(PPFileName.c_str(), 0) == -1)
	//	mkdir(PPFileName.c_str());
	if (access(PPFileName_1.c_str(), 0) == -1)
		mkdir(PPFileName_1.c_str());
// 
	//if (access(NPFileName.c_str(), 0) == -1)
	//	mkdir(NPFileName.c_str());
	//if (access(NNFileName.c_str(), 0) == -1)
	//	mkdir(NNFileName.c_str());


	if (access(palm.c_str(), 0) == -1)
		mkdir(palm.c_str());
	if (access(thumbUp.c_str(), 0) == -1)
		mkdir(thumbUp.c_str());
	if (access(fist.c_str(), 0) == -1)
		mkdir(fist.c_str());
	if (access(neg.c_str(), 0) == -1)
		mkdir(neg.c_str());
	if (access(face.c_str(), 0) == -1)
		mkdir(face.c_str());
	if (access(sideFace.c_str(), 0) == -1)
		mkdir(sideFace.c_str());
	if (access(halfFace.c_str(), 0) == -1)
		mkdir(halfFace.c_str());
	if (access(drape.c_str(), 0) == -1)
		mkdir(drape.c_str());
	if (access(sign.c_str(), 0) == -1)
		mkdir(sign.c_str());
	if (access(stripe.c_str(), 0) == -1)
		mkdir(stripe.c_str());
	if (access(inclineFace.c_str(), 0) == -1)
		mkdir(inclineFace.c_str());

	string strFileListForClassier = "F:\\MyFiles\\Train\\BadFist.txt";

	//string strPosFileListForTest = "E:\\pythonProject\\AllPos\\AllPos1_1.txt";
	string strPosFileListForPalmTest = "E:\\pythonProject\\PosDataForPalmTest0704.txt";
	string strPosFileListForThumbUpTest = "E:\\pythonProject\\PosDataForThumbUpTest0704.txt";
	string strPosFileListForFistTest = "F:\\MyFiles\\Train\\adaFistData.txt";

	string strNegFileListForTest = "E:\\pythonProject\\negTest20180615.txt"; // 1000 samples

	string strNegFileListForThumbUpTest = "D:\\ImageData\\NegRoiSideFaceEar.txt";
	string strNegFileListForPalmTest = "D:\\ImageData\\NegRoiSideFaceEar.txt";
	string strNegFileListForFistTest = "E:\\pythonProject\\NegForFistTest0711.txt";
	
	//CvANN_MLP ann;
	//ann.load("E:\\pythonProject\\Code\\code\\Model\\Zen_PaOnNn21A.xml");
	
#ifdef series_connection
	Ptr<ml::ANN_MLP> ann1 = Algorithm::load<ml::ANN_MLP>("E:\\pythonProject\\Code\\code\\ANNTest\\x64\\Release\\model_lbp_Hog_3layer\\model_lbp_Hog_3layer.xml");//  load model file
	Ptr<ml::ANN_MLP> ann2 = Algorithm::load<ml::ANN_MLP>("E:\\pythonProject\\Code\\code\\ANNTest\\x64\\Release\\palm_tplbp_hog20180127\\palm_tplbp_hog20180127.xml");//  load model file
#else
	Ptr<ml::ANN_MLP> ann = Algorithm::load<ml::ANN_MLP>("F:\\MyFiles\\ANN_Model\\palmModel\\PalmModel_lbp_Hog_3layer_0718_197.xml");//  load model file
#endif
	cout<<"feature extract..."<<endl;
	double dFeatureStartTime=(double)cv::getTickCount();
	Mat matPosForTrain, matNegForTrain, matPosForTest, matNegForTest;
	int itn = 0, itp = 0;
	cout<<"pos samples feature extract"<<endl;
	//ImageFeatureLib::Read_List_Get_LBP_Uniform_Grid( strPosFileListForTest, matPosForTest, true, WIDTH, HEIGHT );
	ImageFeatureLib::Read_List_Get_TPLBP_Hog_Grid(strFileListForClassier, matPosForTest, itp, false, true, 2, 36, 60, HOGBINNUM, HOGGRIDX, HOGGRIDY, LGRIDX, LGRIDY);
	
	cout << "Pos done" << endl;
	
	cout << "neg samples feature extract" << endl;
	//ImageFeatureLib::Read_List_Get_LBP_Uniform_Grid( strNegFileListForTest, matNegForTest, true, WIDTH, HEIGHT );
	//ImageFeatureLib::Read_List_Get_TPLBP_Hog_Grid(strNegFileListForFistTest, matNegForTest, itn, false, true, 2, 36, 36, HOGBINNUM, HOGGRIDX, HOGGRIDY, LGRIDX, LGRIDY);
	cout<<"neg done"<<endl;

	int iPosCountForTest = matPosForTest.rows;
	int iNegCountForTest = matNegForTest.rows;
	double dFeatureEndTime = ( (double)cv::getTickCount() - dFeatureStartTime ) * 1000 / cv::getTickFrequency();
	double dFeatureMeanTime = dFeatureEndTime / ( iPosCountForTest + iNegCountForTest );

	cout<<"test result"<<endl;
	double dPredictStartTime = (double)cv::getTickCount();
	int iPosPos=0, iPosNeg=0, iNegPos=0, iNegNeg=0;
	
	int iPalm = 0;
	int iThumpup = 0;
	int iFist = 0;
	int iNeg = 0;
	int iFace = 0;
	int iSideFace = 0;
	int iHalfFace = 0;
	int iDrape = 0;
	int iSign = 0;
	int iStripe = 0;
	int iInclineFace = 0;

	cout<<"test pos"<<endl;
	for (int iRow=0; iRow<iPosCountForTest; iRow++)
	{
		Mat matTmp = matPosForTest.row(iRow);
		string saveFileName;
		string errSam;
		Mat errImg;
#ifdef series_connection
		Mat matResult1, matResult2;
		ann1->predict( matTmp, matResult1 );
		ann2->predict( matTmp, matResult2 );
		float *pfRow1 = matResult1.ptr<float>(0);
		float *pfRow2 = matResult2.ptr<float>(0);
		if ( pfRow1[0]>pfRow1[1] && pfRow2[0] > pfRow2[1] )
			iPosPos++;
		else
			iPosNeg++;
#else
		Mat matResult;
		ann->predict(matTmp, matResult);
		//float *pfRow = matResult.ptr<float>(0);
		//if (pfRow[0] > pfRow[1])
		float *response = matResult.ptr<float>(0);
		//cout << response << " ";
		if (response[0] > 0.7f)
		{
			iPalm++;
 			saveFileName = getFileName(palm.c_str(), iPalm);
 			errSam = getAnyLine(strFileListForClassier, iRow + 1);
 			errImg = cv::imread(errSam);
 			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: Palm!" << endl;

		}
		else if(response[1] > 0.7f)
		{
			iThumpup++;
			saveFileName = getFileName(thumbUp.c_str(), iThumpup);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: ThumbUp!" << endl;
		}
		else if (response[2] > 0.7f)
		{
			iFist++;
			saveFileName = getFileName(fist.c_str(), iFist);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: Fist!" << endl;
		}
		else if (response[4] > 0.7f)
		{
			iFace++;
			saveFileName = getFileName(face.c_str(), iFace);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: Face!" << endl;
		}
		else if (response[5] > 0.7f)
		{
			iSideFace++;
			saveFileName = getFileName(sideFace.c_str(), iSideFace);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: SideFace!" << endl;
		}
		else if (response[6] > 0.7f)
		{
			iHalfFace++;
			saveFileName = getFileName(halfFace.c_str(), iHalfFace);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: HalfFace!" << endl;
		}
		else if (response[7] > 0.7f)
		{
			iDrape++;
			saveFileName = getFileName(drape.c_str(), iDrape);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: Drape!" << endl;
		}
		else if (response[8] > 0.7f)
		{
			iSign++;
			saveFileName = getFileName(sign.c_str(), iSign);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: Sign!" << endl;
		}
		else if (response[9] > 0.7f)
		{
			iStripe++;
			saveFileName = getFileName(stripe.c_str(), iStripe);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: Stripe!" << endl;
		}
		else if (response[10] > 0.7f)
		{
			iInclineFace++;
			saveFileName = getFileName(inclineFace.c_str(), iInclineFace);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: InclineFace!" << endl;
		}
		else
		{
			iNeg++;
			saveFileName = getFileName(neg.c_str(), iNeg);
			errSam = getAnyLine(strFileListForClassier, iRow + 1);
			errImg = cv::imread(errSam);
			cv::imwrite(saveFileName, errImg);
			cout << "Current samples is: Neg!" << endl;
		}
#endif // series_connection
	}
//	cout << "正样本识别正确数 " << iPosPos << "  " << "正样本识别错误数: " << iPosNeg << endl;
	cout << endl;
	cout<<"test neg"<<endl;
	for (int iRow=0; iRow<iNegCountForTest; iRow++)
	{
		Mat matTmp = matNegForTest.row(iRow);
 		string saveFileName;
 		string errSam;
 		Mat errImg;
#ifdef series_connection

		Mat matResult1, matResult2;
		ann1->predict( matTmp, matResult1 );
		ann2->predict( matTmp, matResult2 );
		float *pfRow1 = matResult1.ptr<float>(0);
		float *pfRow2 = matResult2.ptr<float>(0);
		if ( pfRow1[0]>pfRow1[1] && pfRow2[0] > pfRow2[1] )
			iNegPos++;
		else
			iNegNeg++;
#else
		Mat matResult;
		ann->predict(matTmp, matResult);
		//float *pfRow = matResult.ptr<float>(0);
		//if (pfRow[0] > pfRow[1])
		float *response = matResult.ptr<float>(0);
		//cout << response << " ";
		if (response[0] > 0.7)
		{
			iNegPos++;
// 			saveFileName = getFileName(NPName, iNegPos);
// 			errSam = getAnyLine(strNegFileListForFistTest, iRow + 1);
// 			errImg = cv::imread(errSam);
// 			cv::imwrite(saveFileName, errImg);
		}
		else
		{
			iNegNeg++;
// 			saveFileName = getFileName(NNName, iNegNeg);
// 			errSam = getAnyLine(strNegFileListForTest, iRow + 1);
// 			errImg = cv::imread(errSam);
// 			cv::imwrite(saveFileName, errImg);
		}
#endif // series_connection
	}

	double dPredictEndTime = ( (double)cv::getTickCount() - dPredictStartTime ) * 1000 / cv::getTickFrequency();
	double dPredictMeanTime = dPredictEndTime / ( iPosCountForTest + iNegCountForTest );
	double dSumMeanTime = dFeatureMeanTime + dPredictMeanTime;

	double dfPosPrecision = (double)iPosPos / iPosCountForTest;
	double dfNegPrecision = (double)iNegNeg / iNegCountForTest;
	double dfAllPrecision = (double)(iPosPos + iNegNeg) / (iPosCountForTest + iNegCountForTest);

	cout<<"feature extract mean time:"<<dFeatureMeanTime<<"ms"<<endl;
	cout<<"predict mean time:"<<dPredictMeanTime<<"ms"<<endl;
	cout<<"sum mean time:"<<dSumMeanTime<<"ms"<<endl;
	cout<<endl;
	cout<<"test:"<<"iPosPos: "<<iPosPos<<" "<<"iPosNeg: "<<iPosNeg<<" "<<"iNegPos: "<<iNegPos<<" "<<"iNegNeg: "<<iNegNeg<<endl;
	cout<<"test dfPosPrecision:"<<dfPosPrecision<<endl;
	cout<<"test dfNegPrecision:"<<dfNegPrecision<<endl;
	cout<<"test dfAllPrecision:"<<dfAllPrecision<<endl;
	system("pause");
	return 0;
}

	//load from model file to test 
int main_testImg()
{
	const char* psModelFilePath = "PalmModel_lbp_Hog_3layer0605_big.xml";
	std::string  strSrcImgPath = "iw34738065b373b.jpg";
	cv::Ptr<cv::ml::ANN_MLP> m_PtrAnn = cv::Algorithm::load<cv::ml::ANN_MLP>(psModelFilePath);
	cv::Mat matLayer = m_PtrAnn->getLayerSizes();
	if (matLayer.empty())
	{
		printf("load model file failed\n");
		getchar();
		return -1;
	}
	
	cv::Mat matSrcImg = cv::imread(strSrcImgPath,0);
	if (matSrcImg.empty())
	{
		printf("read src img failed\n");
		getchar();
		return -1;
	}
	imshow("SrcImg", matSrcImg);
	
	cv::Mat matResizedImg;
	//cv::resize(matSrcImg, matResizedImg, cv::Size(IMG_WIDTH, IMG_HEIGHT));
	cv::resize(matSrcImg, matResizedImg, cv::Size(32, 48));
	cv::Mat matFeature;
	matFeature = ImageFeatureLib::Get_TPLBP_Hog_Grid(matResizedImg, 9, 4, 4, 4, 4);
			
	if (matFeature.empty())
	{
		printf("Get TPLBP_HOG_Grid failed\n");
		return -1;
	}
	
	cv::Mat matResult;
	m_PtrAnn->predict(matFeature, matResult);
	assert(matResult.ptr<float>(0) == (float*)matResult.data);
	//float *pRow = (float*)matResult.data;
	float *pfRow = matResult.ptr<float>(0);
		
	
	printf("palm:%f\n", pfRow[0]);
	printf("negpalm:%f\n", pfRow[1]);
	//printf("face:%f\n", pfRow[2]);
	//printf("side face:%f\n", pfRow[3]);
	//printf("half face:%f\n", pfRow[4]);
	//printf("drape:%f\n", pfRow[5]);
	//printf("sign:%f\n", pfRow[6]);
	//printf("stripe:%f\n", pfRow[7]);
	//printf("thumbup:%f\n", pfRow[8]);
	
	
	cv::waitKey();
	getchar();
	//if (pRow[0] > pRow[1])
	//	return 1; // match
	return 0; // mismatch
		
}
string getAnyLine(string filename, int index)
{
	string line;
	ifstream in(filename);
	int count = 0;
	if (in)
	{
		while (getline(in, line))
		{
			count++;
			if (count == index)
			{
				//cout << line << endl;
				return line;
			}
		}

	}
	else
	{
		cout << " Read File Error" << endl;
		return NULL;
	}
}

string getFileName(const char* Name, int num)
{
	SYSTEMTIME st;
	CString strDate, strTime;
	GetLocalTime(&st);
	int year = st.wYear;
	int month = st.wMonth;
	int day = st.wDay;
	int hour = st.wHour;
	int minute = st.wMinute;
	int second = st.wSecond;

	char timeName[30];
	char dstFileName[200];
	sprintf_s(timeName, "%04d%02d%02d%02d%02d%02d%s", year, month, day, hour, minute, second, "_");
	sprintf_s(dstFileName, "%s/%s%d%s", Name, timeName, num, ".jpg");
	return dstFileName;
}