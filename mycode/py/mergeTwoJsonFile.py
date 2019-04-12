import json

JSON_File_in_1 = "F:\\caffe-windows\\cocodata\\train.json"
JSON_File_in_2 = "F:\\caffe-windows\\cocodata\\ai_challenger_train.json"
JSON_File_out = "F:\\caffe-windows\\cocodata\\aicoco_train.json"


with open(JSON_File_in_1, 'r') as rf1:
    s1 = json.load(rf1)
    with open(JSON_File_in_2, 'r') as rf2:
        s2 = json.load(rf2)
        with open(JSON_File_out, 'w') as wf:
            for key in s2.keys(): # annotations categories images
                # print(key)
                if key == 'annotations':  # add subDirectory
                    i = 700000
                    j = 0
                    for data in s2[key]:
                        # print(s1[key][len(s1[key]) - 1])
                        s2[key][j]['id'] = i
                        s2[key][j]['id'] = i
                        s2['images'][j]['id'] = i
                        s1[key].append(s2[key][j])
                        s1['images'].append(s2['images'][j])
                        # s1[key][i]['file_name'] = valPath + data['file_name'] # here to valPath or trainPath
                        # print(str(i) + " :" + s1[key][i]['file_name'])
                        i += 1
                        j += 1
                        print("process: " + str(j))
            json.dump(s1, wf)
print("complete!")