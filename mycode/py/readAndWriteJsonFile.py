import json

JSON_File_in = "F:\\caffe-windows\\cocodata\\annotations\\person_keypoints_val2017.json"
JSON_File_out = "F:\\caffe-windows\\cocodata\\annotations\\val.json"

trainPath = "train2017/"
valPath = "val2017/"
with open(JSON_File_in, 'r') as rf:
    with open(JSON_File_out, 'w') as wf:
        s = json.load(rf)
        for key in s.keys(): # annotations licenses info categories images
            if key == 'images':  # add subDirectory
                print(key)
                i=0
                for data in s[key]:
                    if len(data) == 8:
                        s[key][i]['file_name'] = valPath + data['file_name'] # here to valPath or trainPath
                        print(str(i) + " :" + s[key][i]['file_name'])
                        i += 1
            elif key == 'categories':  # add subDirectory
                print(key)
                categoriesKeyPoint = ["Top","Neck","RShoulder","RElbow","RWrist","LShoulder","LElbow","LWrist","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle"]
                s[key][0]['keypoints'] = categoriesKeyPoint
            elif key == 'annotations': # convert the coco 17 points to AI chanllger 14 points
                print(key)
                j = 0
                newKeyPointOrder = []
                for data in s[key]:
                    top_x = int((int(data['keypoints'][3]) + int(data['keypoints'][6]))/2)
                    newKeyPointOrder.append(top_x)
                    # top_y = int((int(data['keypoints'][4]) + int(data['keypoints'][7]))/2)
                    top_y=int(2*(int(data['keypoints'][4])-int(data['keypoints'][1])))
                    newKeyPointOrder.append(top_y)
                    top_flag=data['keypoints'][2]
                    # top_flag = 2 # this maybe uncorrect
                    newKeyPointOrder.append(top_flag)

                    neck_x = int((int(data['keypoints'][15]) + int(data['keypoints'][18]))/2)
                    newKeyPointOrder.append(neck_x)
                    # neck_y = int((int(data['keypoints'][16]) + int(data['keypoints'][19]))/2)
                    neck_y1 = int((int(data['keypoints'][16]) + int(data['keypoints'][1]))/2)
                    neck_y2 = int((int(data['keypoints'][19]) + int(data['keypoints'][1]))/2)
                    neck_y=int((neck_y1+neck_y2)/2)
                    newKeyPointOrder.append(neck_y)
                    neck_flag=data['keypoints'][5]
                    # neck_flag = 2 # this maybe uncorrect
                    newKeyPointOrder.append(neck_flag)


                    RShoulder = data['keypoints'][18:21]
                    for i in range(len(RShoulder)):
                        newKeyPointOrder.append(RShoulder[i])

                    RElbow = data['keypoints'][24:27]
                    for i in range(len(RElbow)):
                        newKeyPointOrder.append(RElbow[i])

                    RWrist = data['keypoints'][30:33]
                    for i in range(len(RWrist)):
                        newKeyPointOrder.append(RWrist[i])


                    LShoulder = data['keypoints'][15:18]
                    for i in range(len(LShoulder)):
                        newKeyPointOrder.append(LShoulder[i])

                    LElbow = data['keypoints'][21:24]
                    for i in range(len(LElbow)):
                        newKeyPointOrder.append(LElbow[i])

                    LWrist = data['keypoints'][27:30]
                    for i in range(len(LWrist)):
                        newKeyPointOrder.append(LWrist[i])



                    RHip = data['keypoints'][36:39]
                    for i in range(len(RHip)):
                        newKeyPointOrder.append(RHip[i])

                    RKnee = data['keypoints'][42:45]
                    for i in range(len(RKnee)):
                        newKeyPointOrder.append(RKnee[i])

                    RAnkle = data['keypoints'][48:]
                    for i in range(len(RAnkle)):
                        newKeyPointOrder.append(RAnkle[i])


                    LHip = data['keypoints'][33:36]
                    for i in range(len(LHip)):
                        newKeyPointOrder.append(LHip[i])

                    LKnee = data['keypoints'][39:42]
                    for i in range(len(LKnee)):
                        newKeyPointOrder.append(LKnee[i])

                    LAnkle = data['keypoints'][45:48]
                    for i in range(len(LAnkle)):
                        newKeyPointOrder.append(LAnkle[i])

                    s[key][j]['keypoints'] = newKeyPointOrder
                    print("%d complete!" % j)
                    newKeyPointOrder = []
                    j += 1
        json.dump(s, wf)
print("complete!")





