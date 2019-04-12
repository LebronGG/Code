#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <atltime.h>
#include <windows.h>
#include <stdlib.h>
#include <io.h>
#include <direct.h>
#include <fstream>
using namespace cv;
using namespace std;
#define INPUT_TXT_FILE        "E:\\videotest\\.txt"
#define StorageFile     "E:\\opencvtest\\videowrite\\temp"

int main(int argc, char**argv)
{
	const string storagefile = StorageFile;
	SYSTEMTIME st;
	char imageName[200];
	char timeName[30];
	string strFileName;
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
			VideoCapture capture(strFileName);
			while (1)
			{
				Mat frame;
				capture >> frame;
				if (frame.empty())
				{
					cout << "Process Current Video Complete!" << endl;
					break;
				}
				GetLocalTime(&st);
				int framecount = 0;
				int year = st.wYear;
				int month = st.wMonth;
				int day = st.wDay;
				int hour = st.wHour;
				int minute = st.wMinute;
				int second = st.wSecond;
				sprintf_s(timeName, "%04d%02d%02d%02d%02d%02d%s", year, month, day, hour, minute, second, "_");
				sprintf_s(imageName, "%s/%s%d%s", storagefile.c_str(), timeName, ++framecount, ".jpg");
				//imwrite(imageName, frame);//保存保存一帧图片   
				cout << "当前帧: " << framecount << endl;
			}
		}
		else
			{
				continue;
			}
	}
	system("pause");
	return 0;
}
//跳帧保存
#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include <atltime.h>
#include <windows.h>
#include <stdlib.h>
#include <io.h>
#include <direct.h>
#include <fstream>
using namespace cv;
using namespace std;
#define INPUT_TXT_FILE        "D:\\cbg.txt"
#define StorageFile     "E:\\opencvtest\\videowrite\\temp"

int main(int argc, char**argv)
{
	const string storagefile = StorageFile;
	SYSTEMTIME st;
	char imageName[200];
	char timeName[30];
	string strFileName;
	int framecount = 0;
	int count = 0;
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
			VideoCapture capture(strFileName);
			while (1)
			{
				Mat frame;
				capture >> frame;

				if (frame.empty())
				{
					cout << "Process Current Video Complete!" << endl;
					break;
				}
				GetLocalTime(&st);
				int year = st.wYear;
				int month = st.wMonth;
				int day = st.wDay;
				int hour = st.wHour;
				int minute = st.wMinute;
				int second = st.wSecond;
				sprintf_s(timeName, "%04d%02d%02d%02d%02d%02d%s", year, month, day, hour, minute, second, "_");
				sprintf_s(imageName, "%s/%s%d%s", storagefile.c_str(), timeName, ++framecount, ".jpg");
				if (framecount % 25 == 0)
				{
					cout << "正在写第" << framecount << "帧" << endl;
					imwrite(imageName, frame);//保存保存一帧图片  
					//imshow("video", frame);
					count++;
					cout << "存入照片的个数为:" << count << endl;
				}
			}
		}
	}
	system("pause");
	return 0;
}


