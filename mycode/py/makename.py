import os
import shutil
from PIL import Image
srcPath=r'E:\BaiduYunDownload\WIDER_train\WIDER_test\images'
count=0
fo=open("filevideo.txt","a+")
for root,dirs,files in os.walk(srcPath):
    for name in files:
        count=count+1
        srcCompleteName=os.path.join(root,name)
        print("count: " + str(count) + " " + srcCompleteName)
        fo.writelines(srcCompleteName+'\n')
fo.close()
print('complete')
