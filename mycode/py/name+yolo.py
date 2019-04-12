import os
import shutil
from PIL import Image
srcPath=r'E:\opencvtest\adaAndANN\JPEGImages'
label_file='E:/opencvtest/adaAndANN/labels'
count=0
fo=open("train.txt","a+")
for root,dirs,files in os.walk(srcPath):
    for name in files:
        srcCompleteName=os.path.join(root,name) 
        fo.writelines(srcCompleteName+'\n')
        
        name=name.split(".")
        name=name[0]
        list_file = open(label_file+'/%s.txt' % (name), 'w')
        list_file.write('0 0.5 0.5 0.99 0.99')
        list_file.close()
        count=count+1        
        print("count: " + str(count) + " " + srcCompleteName)
fo.close()
print('complete')
