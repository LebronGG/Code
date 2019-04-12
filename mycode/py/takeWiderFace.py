import cv2

mainFolder = 'WIDER_train/images/'

# label file
txtFileName = 'wider_face_train_bbx_gt.txt'

# out file
outFileName = 'labels.txt'
outImgFolder = 'F:/MyFiles/papers/MTCNN/caffeCode/mtcnn-master/train/samples/'
count = 0
numPic = 0

with open(outFileName, 'a+') as fw:
    with open(txtFileName, 'r') as fr:
        picLine = fr.readline()
        # print(line)
        while picLine:
            count += 1
            picLineSplit = picLine.strip().split("/")
            endStr = picLineSplit[-1].split(".")[-1]
            if (endStr == 'jpg'):
                numFaceLine = fr.readline()
                if (int(numFaceLine) == 1):
                    numPic += 1
                    img = cv2.imread(mainFolder + picLine)
                    # cv2.imwrite(outImgFolder + picLineSplit[-1], img)
                    for i in range(int(numFaceLine)):
                        gtline = fr.readline()
                        gtline = gtline.strip().split(' ')
                        fw.writelines('samples/' + picLineSplit[-1] + ' ' + gtline[0] + ' ' + gtline[1] + ' '
                        + str(int(gtline[2]) + int(gtline[0])) + ' ' + str(int(gtline[1]) + int(gtline[3])) + '\n')
                        print('number Faces: ' + str(numPic))
                        # print(gtline)
            picLine = fr.readline()

            # img = cv2.imread(attr[0])
            # for i in range(int(attr[1])):
            #     imgRoi = img[int(attr[4*i+3]):(int(attr[4*i+3]) + int(attr[4*i+5])),
            #                  int(attr[4*i+2]):(int(attr[4*i+2]) + int(attr[4*i+4]))]

    print('Process complete !')
