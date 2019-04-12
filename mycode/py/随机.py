with open('F:\\dataset\\Bottom\\1.txt') as f:
        content = f.readlines()
 
print(content)

import random 
 
random.shuffle(content) #随机排序
print(content)

match_image_random = open("F:\\dataset\\Bottom\\2.txt","w")
for i in range(778042):
    match_image_random.write(content[i])
