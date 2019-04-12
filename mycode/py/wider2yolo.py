import cv2

filename = 'E:\BaiduYunDownload\WIDER_train\wider_face_test.txt'
imge_file='E:/BaiduYunDownload/WIDER_train/WIDER_test/images'
yolo_label_file='E:/py/darknet/build/darknet/x64/VOCdevkit/VOC2007/labels'
yolo_image_file='E:/py/darknet/build/darknet/x64/VOCdevkit/VOC2007/JPEGImages'

def convert(size, box):                 #size是图片尺寸 box是坐标
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def readfile(filename):
    fo = open(filename, "r")  # 读入标注结果集
    while True:
        key = next(fo, -1)
        if key == -1:
            break;
        key = key.replace('\n', '')
        key1 = key.split("/")
        key1 = key1[1].split(".")
        key1 = key1[0]         #获取图片名称
        list_file = open(yolo_label_file+'/%s.txt' % (key1), 'w')                                             #新建对应图片的label，存放在My_labels文件夹下
        value = []
        key = imge_file+'/%s'%(key) 	#该图片位置
        # print(key)
        image = cv2.imread(key)
        cv2.imwrite(yolo_image_file+'/%s.jpg'%(key1), image)
        image_size = []
        # print(image.shape[0],image.shape[1])
        image_size = [image.shape[1],image.shape[0]]
        num = next(fo, -1)
        for i in range(int(num)):
            value = next(fo, -1).split(' ')
            box = [int(value[0]),int(value[0])+int(value[2]),int(value[1]),int(value[1])+int(value[3])]
            x, y, w, h = convert(image_size,box)
            # print(x, y, w, h)
            list_file.write('0 %s %s %s %s\n' % (x,y,w,h))

    fo.close()
readfile(filename)