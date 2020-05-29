import os
folderName = './'
#获取指定路径的所有文件名字
dirList = os.listdir(folderName)
print(dirList)
n = len(dirList)
l = 0
k = 0
m = 0
#遍历输出所有文件名字
for i in range(0,n):
	if dirList[i][-1] == 'g':
		newName = str(l) + ".jpg"
		os.rename(folderName+dirList[i], folderName+newName)
		l = l + 1
	elif dirList[i][-1] == 't':
		newName = str(k) + '.txt'
		os.rename(folderName+dirList[i], folderName+newName)
		k = k + 1
	elif dirList[i][-1] == 'l':
		newName = str(m) + '.xml'
		os.rename(folderName+dirList[i], folderName+newName)
		m = m + 1
	else:
		break
	
