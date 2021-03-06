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


int main(int argc, char** argv)
{
	const string faces = "./face";
	if (_access(faces.c_str(), 0) == -1)
		_mkdir(faces.c_str());
	float radio_img = 1.0;
	float radio_ROI_y = 0.1;
	float radio_ROI_h = 1;
	SYSTEMTIME st;
	char imageName[200];
	char timeName[30];
	string strFileName;
	int framecount = 0;
	frontal_face_detector detector = get_frontal_face_detector();
	VideoCapture cap(0);
	while (1)
	{
		Mat frame, faceROI, tmpImg;
		int x = 0, y = 0, w = 0, h = 0; // the dlib face location
		int ROI_x = 0, ROI_y = 0, ROI_w = 0, ROI_h = 0;
		cap >> frame;
		frame.copyTo(tmpImg);
		dlib::cv_image<bgr_pixel> img(frame);
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
			cv::rectangle(frame, Rect(ROI_x, ROI_y, ROI_w, ROI_h), Scalar(255, 0, 100), 2); // rectangle the detect ROI
			GetLocalTime(&st);
			int year = st.wYear;
			int month = st.wMonth;
			int day = st.wDay;
			int hour = st.wHour;
			int minute = st.wMinute;
			int second = st.wSecond;
			sprintf_s(timeName, "%04d%02d%02d%02d%02d%02d%s", year, month, day, hour, minute, second, "_");
			sprintf_s(imageName, "%s/%s%d%s", faces.c_str(), timeName, ++framecount, ".jpg");
			imwrite(imageName, faceROI);//每一帧的随机保存
			cout << "采集个数：" << framecount << endl;
			if (framecount > 999)
				framecount = 0;
		}
		imshow("video", frame);
		int c = cv::waitKey(10);
		if (c == 27)
			return -1;
	}
	return 0;
}