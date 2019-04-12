#include "objdetect/objdetect.hpp"
#include <opencv2/objdetect/objdetect_c.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/opencv.hpp"
#include "ImageFeatrueExtract.h"

#include <atltime.h>
#include <windows.h>
#include <stdlib.h>
#include <io.h>
#include <direct.h>

#include <fstream>
using namespace cv;
using namespace ml;
using namespace std;

#define ANN_MODEL_FILE_Large                              "./modelFile/faceModel_lbp_Hog_3layer0825_60X60.xml"
#define ANN_MODEL_FILE_Small                              "./modelFile/faceModel_lbp_Hog_3layer0825_35X35.xml"
#define ANN_MODEL_FILE_Single                             "./modelFile/faceModel_lbp_Hog_3layer0831_40X40.xml"
#define INPUT_TXT_FILE                                    "./data/faceNeg3.txt"
#define OUTPUT_TXT_FILE                                   "./falseData/falseAlarm.txt"
//#define Large_Small_Ann
int main(int argc, char** argv)
{
	// store image
	bool savePosFace = false;
	bool saveNegFace = false;
	float face_thresh = 0.66f;
	int face_wh_thresh = 40;
	float prediction_radio = 0.5;
	float score = 0;
	const string PalmFileName = "./AdaFace";
	const string AnnRoiFilePalm = "./AnnFace";

	char imageName[200];
	char AnnImageName[200];
	char AnnRoiImageName[200];
	char timeName[30];
	int index = 0;
	int iTrueTrue = 0, iTrueFalse = 0, iCount = 0;
	bool saveROI = true;

	if (saveNegFace)
	{
		if (_access(PalmFileName.c_str(), 0) == -1)
			_mkdir(PalmFileName.c_str());
	}
	if (savePosFace)
	{
		if (_access(AnnRoiFilePalm.c_str(), 0) == -1)
			_mkdir(AnnRoiFilePalm.c_str());
	}
	SYSTEMTIME st;
	CString strDate, strTime;

#ifdef Large_Small_Ann

	Ptr<ANN_MLP> ann_Large = Algorithm::load<ANN_MLP>(ANN_MODEL_FILE_Large); // load ANN model File 
	if (ann_Large->empty())
	{
		std::cout << "Load File: " << ANN_MODEL_FILE_Large << " Error" << std::endl;
		getchar();
		return -1;
	}

	Ptr<ANN_MLP> ann_Small = Algorithm::load<ANN_MLP>(ANN_MODEL_FILE_Small); // load ANN model File 
	if (ann_Small->empty())
	{
		std::cout << "Load File: " << ANN_MODEL_FILE_Small << " Error" << std::endl;
		getchar();
		return -1;
	}
#else
	Ptr<ANN_MLP> ann = Algorithm::load<ANN_MLP>(ANN_MODEL_FILE_Single); // load ANN model File 
	if (ann->empty())
	{
		std::cout << "Load File: " << ANN_MODEL_FILE_Single << " Error" << std::endl;
		getchar();
		return -1;
	}
#endif // Large_Small_Ann

#ifdef INPUT_TXT_FILE
	ifstream inStr(INPUT_TXT_FILE);
	if (!inStr.is_open())
	{
		getchar();
		return -1;
	}
	ofstream outStr(OUTPUT_TXT_FILE);
#else
	VideoCapture cap(0);
#endif // INPUT_TXT_FILE


#ifdef INPUT_TXT_FILE
	std::string strFileName;
	std::cout << "Processing :" << INPUT_TXT_FILE << std::endl;
#endif // INPUT_TXT_FILE

	cv::Mat frame_cpp, grayFrame;
	
	while (std::getline(inStr, strFileName))
	{
		if (!strFileName.empty())
		{
			frame_cpp = imread(strFileName);
		}
		
		//capture >> frame_cpp;
		if (frame_cpp.empty())
		{
			std::cout << "Read frame Error,File Name:" << strFileName << std::endl;
			continue;
		}
		std::cout << "Count: " << ++iCount << ", Process File: " << strFileName << std::endl;
		if (frame_cpp.channels() == 3)
			cvtColor(frame_cpp, grayFrame, COLOR_BGR2GRAY);
		else
			grayFrame = frame_cpp;
#ifdef Large_Small_Ann

		if (frame_cpp.cols > face_wh_thresh) 
		{
			cv::resize(grayFrame, grayFrame, Size(60, 60), INTER_CUBIC);
		}
		else
		{
			cv::resize(grayFrame, grayFrame, Size(36, 36), INTER_CUBIC);
		}
		cv::Mat feature = ImageFeatureLib::Get_TPLBP_Hog_Grid(grayFrame, 9, 4, 4, 4, 4);
		cv::Mat result_large, result_small;
		ann_Large->predict(feature, result_large);
		ann_Small->predict(feature, result_small);
		float* pfRow_large = result_large.ptr<float>(0);
		float* pfRow_small = result_small.ptr<float>(0);
		if (frame_cpp.cols >= face_wh_thresh)
		{
			score = prediction_radio * pfRow_large[0] + (1 - prediction_radio) * pfRow_small[0];
		}
		else
		{
			score = (1 - prediction_radio) * pfRow_large[0] + prediction_radio * pfRow_small[0];
		}
#else
		cv::resize(grayFrame, grayFrame, Size(60, 60), INTER_CUBIC);
		cv::Mat feature = ImageFeatureLib::Get_TPLBP_Hog_Grid(grayFrame, 9, 4, 4, 4, 4);
		cv::Mat result;
		ann->predict(feature, result);
		float* pfRow = result.ptr<float>(0);
		float score = pfRow[0];
		cout << " Score: " << score << endl;
		outStr << strFileName << " Score: " << score << endl;
#endif // INPUT_TXT_FILE
		if (score > face_thresh)
		{
			iTrueTrue++;
			//outStr << strFileName << " Score: " << score <<endl;
		}
		else
		{
			iTrueFalse++;
			//outStr << strFileName << " Score: " << score << endl;
		}
		waitKey(0);
		strFileName.clear();
	}
	double accuracy = (iTrueTrue / (double)iCount);
	double falseAccuracy = (iTrueFalse / (double)iCount);
	std::cout << "True Number: " << iTrueTrue << ", Accuracy: " << accuracy << std::endl;
	std::cout << "False Number: " << iTrueFalse << ", False radio: " << falseAccuracy << std::endl;
	
	inStr.close();
	outStr.close();
	system("pause");
	return 0;
}