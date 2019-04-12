import os
import shutil
from PIL import Image
srcPath=r'E:\workplace\boxdata1\neg'
count=0
fo=open("neg.txt","a+")
for root,dirs,files in os.walk(srcPath):
    for name in files:
        count=count+1
        #if(count%1000==0):
        #    print(count)
        srcCompleteName=os.path.join(root,name)
        #print(srcCompleteName+'\n')
        img = Image.open(srcCompleteName)
        #imagesize = img.size
        #print(imagesize)
        #print(img.size[0])
        #print(img.size[1])
        #srcCompleteName = srcCompleteName 
        print("count: " + str(count) + " " + srcCompleteName)
        fo.writelines(srcCompleteName+'\n')
fo.close()
print('complete')
