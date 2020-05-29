import os
import cv2
folderName = './'
#获取指定路径的所有文件名字
dirList = os.listdir(folderName)
n = len(dirList)
#遍历输出所有文件名字
for i in range(0,n):
	if dirList[i][-1] == "g":
		img = cv2.imread(dirList[i])
		img_tran = cv2.flip(img, 1)
		cv2.imwrite(str(i + 151) + "_windows.png",img_tran)