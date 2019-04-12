import os

class BatchRename():
    def __init__(self):
        self.path = 'C:/Users/Zienon/Desktop/新建文件夹/1/'

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)
        i = 100000
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), str(i) + '.jpg')
                try:
                    os.rename(src, dst)
                    print ('converting %s to %s ...' % (src, dst))
                    i = i + 1
                except:
                    continue

if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()