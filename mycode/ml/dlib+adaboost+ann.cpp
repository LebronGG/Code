#include "objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/opencv.hpp"
#include "ImageFeatrueExtract.h"
#include <dlib\image_processing\frontal_face_detector.h>
#include <dlib\gui_widgets.h>
#include <dlib\image_io.h>
#include<dlib\opencv.h>
#include<iostream>

#include <atltime.h>
#include <windows.h>
#include <stdlib.h>
#include <io.h>
#include <direct.h>
#include <fstream>

using namespace dlib;
using namespace cv;
using namespace ml;
using namespace std;

#define ADABOOST_MODEL_FILE						"./modelFile/xuModelFile_1KX2K_17.xml"
#define ANN_MODEL_FILE                              "./modelFile/xuModel_TPLBP_Hog_3layer0807_99.xml"


int main(int argc, char** argv)
{
	const string faces = "./temp";
	if (_access(faces.c_str(), 0) == -1)
		_mkdir(faces.c_str());
	float radio_img = 1.0;
	float radio_ROI_y = 0.1;
	float radio_ROI_h = 1.5;
	Ptr<ANN_MLP> ann = Algorithm::load<ANN_MLP>(ANN_MODEL_FILE); // load ANN model File
	if (ann->empty())
	{
		std::cout << "Load ANN model File Error" << std::endl;
		getchar();
		return -1;
	}
	CascadeClassifier object_cascade;
	object_cascade.load(ADABOOST_MODEL_FILE);  // load  Adaboost file
	if (object_cascade.empty())
	{
		std::cout << " Load File Failed" << std::endl;
		getchar();
		return -1;
	}

	VideoCapture cap(0);
	frontal_face_detector detector = get_frontal_face_detector();

	while (1)
	{
		Mat frame, faceROI, tmpImg, AdaImg, AnnImg, GrayfaceROI, resizeImg;
		int x = 0, y = 0, w = 0, h = 0; // the dlib face location
		int ROI_x = 0, ROI_y = 0, ROI_w = 0, ROI_h = 0;
		cap >> frame;
		frame.copyTo(tmpImg);
		frame.copyTo(AdaImg);
		frame.copyTo(AnnImg);

		cv::resize(frame, resizeImg, Size(frame.cols * radio_img, frame.rows * radio_img)); //resize the source image
		//cout << "Resize img Width: " << resizeImg.cols << " , Height: " << resizeImg.rows << endl;
		dlib::cv_image<bgr_pixel> img(resizeImg);
		//cout << "Resize img Width: " << img.nc() << " , Height: " << img.nr() << endl;
		std::vector<dlib::rectangle> dets = detector(img);
		cout << "Face Num: " << dets.size() << endl;;
		for (int i = 0; i < dets.size(); i++)
		{
			x = dets[i].left();
			y = dets[i].top();
			w = dets[i].right() - dets[i].left();
			h = dets[i].bottom() - dets[i].top();

			if (x < 0)
				x = 0;
			if (y < 0)
				y = 0;
			if (x + w > img.nc())
				w = img.nc() - x - 1;
			if (y + h > img.nr())
				h = img.nr() - y - 1;

			ROI_x = (1 / radio_img) * x;
			if (ROI_x < 0)
				ROI_x = 0;

			ROI_y = (1 / radio_img) * y + y * radio_ROI_y;
			if (ROI_y < 0)
				ROI_y = 0;

			ROI_w = (1 / radio_img) * w;
			if (ROI_x + ROI_w > tmpImg.cols)
			{
				ROI_w = tmpImg.cols - ROI_x - 1;
				cout << "First: ROI_x: " << ROI_x << ", ROI_w: " << ROI_w << ", ROI_y: " << ROI_y << ", ROI_h: " << ROI_h << endl;
			}
			ROI_h = (1 / radio_img) * h * radio_ROI_h;
			if ((ROI_h + ROI_y) > tmpImg.rows)
			{
				ROI_h = tmpImg.rows - ROI_y - 1;
				cout << "Second: ROI_x: " << ROI_x << ", ROI_w: " << ROI_w << ", ROI_y: " << ROI_y << ", ROI_h: " << ROI_h << endl;
			}
			faceROI = tmpImg(Rect(ROI_x, ROI_y, ROI_w, ROI_h));
			cv::rectangle(AdaImg, Rect(ROI_x, ROI_y, ROI_w, ROI_h), Scalar(255, 0, 100), 2); // rectangle the detect ROI

			if (faceROI.channels() == 3)
				cvtColor(faceROI, GrayfaceROI, COLOR_BGR2GRAY);
			else
				GrayfaceROI = faceROI;
			std::vector<cv::Rect> vecObjects;
			object_cascade.detectMultiScale(GrayfaceROI, vecObjects, 1.1, 1, 0, Size(18, 30), Size(144, 240));
			Mat temp;
			faceROI.copyTo(temp);
			for (unsigned int i = 0; i < vecObjects.size(); i++)
			{
				Rect rect1 = vecObjects[i];
				int x1, y1, width, height;
				x1 = (int)(rect1.x - (rect1.width) * 0.0);
				y1 = (int)(rect1.y - (rect1.height) * 0.0);
				width = (int)(rect1.width) * 1.0;
				height = (int)(rect1.height) * 1.0;
				if (x1 < 0)
					x1 = 0;
				if (y1 < 0)
					y1 = 0;
				if ((x1 + width) > temp.cols)
				{
					width = (int)(temp.cols) - x1 - 1;
					cout << "First: x1: " << x1 << ", w: " << width << ", y1: " << y1 << ", h: " << height << endl;
				}
				if ((y1 + height) > temp.rows)
				{
					height = (int)(temp.rows) - y1 - 1;
					cout << "Second: x1: " << x1 << ", w: " << width << ", y1: " << y1 << ", h: " << height << endl;
				}
				cv::Mat imageROI = temp(Rect(x1, y1, width, height));

				cv::Mat extractRoi;
				if (imageROI.channels() == 3)
					cv::cvtColor(imageROI, extractRoi, COLOR_BGR2GRAY);
				else
					extractRoi = imageROI;
				cv::rectangle(AdaImg, Rect(ROI_x + x1, ROI_y + y1, width, height), Scalar(0, 0, 255), 2);

				cv::resize(extractRoi, extractRoi, Size(36, 60), INTER_CUBIC);
				cv::Mat feature = ImageFeatureLib::Get_TPLBP_Hog_Grid(extractRoi, 9, 4, 4, 4, 4);
				cv::Mat result;
				ann->predict(feature, result);
				float* pfRow = result.ptr<float>(0);
				if (pfRow[0] > 0.5)
				{
					cv::rectangle(AnnImg, Rect(ROI_x + x1, ROI_y + y1, width, height), Scalar(0, 255, 0), 2);
				}
			}
		}
		cv::imshow("Adaboost", AdaImg);//adaboostœ‘ æ
		cv::imshow("Ann", AnnImg);//annœ‘ æ
		int c = waitKey(1);
		if (c == 27)
			break;
	}
	destroyAllWindows();
	return 0;
}




