import cv2
import os

mainFolder = 'WIDER_train/images/'

# label file
txtFileName = 'wider_face_train_bbx_gt.txt'

# out file
outFileName = 'E:/BaiduYunDownload/WIDER_train/label/'
outImgFolder = 'F:/caffe-windows/darknet-gpu/build/darknet/x64/VOCdevkit/VOC2007/JPEGImages/'
count = 0
numPic = 0

with open(txtFileName, 'r') as fr:
    picLine = fr.readline()
    while picLine:
        count += 1
        picLineSplit = picLine.strip().split("/")
        endStr = picLineSplit[-1].split(".")[-1]
        filename = picLineSplit[-1].split(".")[-2]
        if (endStr == 'jpg'):
            out_file = open('labels/%s.txt'%filename, 'w')
            numFaceLine = fr.readline()
            numPic += 1
            img = cv2.imread((mainFolder + picLine).strip('\n'))
            h=img.shape[0]
            w=img.shape[1] 
            cv2.imwrite(outImgFolder + picLineSplit[-1], img)
            for i in range(int(numFaceLine)):
                gtline = fr.readline()
                gtline = gtline.strip().split(' ')
                x=int(gtline[0])
                y=int(gtline[1])
                width=int(gtline[2])
                height=int(gtline[3])
                out_file.writelines('0 ' +str((2*x+width)/(2*w)) + ' ' + str((2*y+height)/(2*h)) + ' '+ str((width/w)) + ' ' + str((height/h)) + '/n')
            out_file.close()
        picLine = fr.readline()
        print('图片个数',count)
    print('Process complete !')
