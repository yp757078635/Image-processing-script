import os
import cv2
folderName = './'
img=cv2.imread(folderName)
print(img)
dirList = os.listdir(folderName)
print(dirList)

for name in dirList:
	if name[-1] == 'g':
		img = cv2.imread(name)
		img_resize = cv2.resize(img,(448,448))
		cv2.imwrite(name,img_resize)