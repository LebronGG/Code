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
#include <stdio.h>

using namespace dlib;
using namespace cv;
using namespace ml;
using namespace std;

#define INPUT_TXT_FILE        "C:\\Users\\Administrator\\Desktop\\faceneg\\face.txt"
#define StorageFile           "C:\\Users\\Administrator\\Desktop\\faceneg\\face"

int main(int argc, char** argv)
{
	const string storagefile = StorageFile;
	SYSTEMTIME st;
	char imageName[200];
	char timeName[30];
	string strFileName;
	char remove_file[200];
	frontal_face_detector detector = get_frontal_face_detector();
	int x1, y1, x2, y2;
	int framecount = 0;
	ifstream inStr(INPUT_TXT_FILE);
	if (!inStr.is_open())
	{
		cout << "read videofile error" << endl;
		getchar();
		return -1;
	}
	while (std::getline(inStr, strFileName))
	{
		if (!strFileName.empty())
		{
			Mat frame = imread(strFileName);
			if (frame.empty())
			{
				cout << "Process Current Video Complete!" << endl;
				continue;
			}
			dlib::cv_image<rgb_pixel> img(frame);
			std::vector<dlib::rectangle> dets = detector(img);
			cout << "存在目标个数：" << dets.size() << endl;
			if (!dets.empty())
			{
				for (int i = 0; i < dets.size(); i++)
				{
					x1 = dets[i].left();
					y1 = dets[i].top();
					x2 = dets[i].right();
					y2 = dets[i].bottom();
					if (x1 < 0)
						x1 = 0;
					if (y1 < 0)
						y1 = 0;
					if (x2 > frame.cols)
						x2 = frame.cols;
					if (y2 > frame.rows)
						y2 = frame.rows;
					Rect rect(x1, y1, x2 - x1, y2 - y1);
					cv::rectangle(frame, rect, Scalar(255, 0, 0), 1, 1, 0);
					GetLocalTime(&st);
					int year = st.wYear;
					int month = st.wMonth;
					int day = st.wDay;
					int hour = st.wHour;
					int minute = st.wMinute;
					int second = st.wSecond;
					Mat imgROI = frame(rect);
					sprintf_s(timeName, "%04d%02d%02d%02d%02d%02d%s", year, month, day, hour, minute, second, "_");
					sprintf_s(imageName, "%s/%s%d%s", storagefile.c_str(), timeName, ++framecount, ".jpg");
					imwrite(imageName, imgROI);
					cout << "采样人脸个数：" << framecount << endl;
					if (framecount > 999)
						framecount = 0;
				}
				strcpy(remove_file, strFileName.c_str());
				cout << "删除路径：" << remove_file << endl;
				std::remove(remove_file);
				int result = std::remove(remove_file);
				if (result == 0)
					cout << "delete succeeded!" << endl;
				else
					cout << "delete failed!" << endl;
			}
			else
			{
				continue;
			}
		}
	}
system("pause");
return 0;
}

