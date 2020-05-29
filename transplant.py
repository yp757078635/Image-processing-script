import cv2
import numpy as np
import os
import random
class Superimpose:

	def __init__(self):
		self.number_photo = 7 # 每个背景生成多少张图片
		self.sample_item = 4 # 每一个目标在图片中的最大个数
		self.random_resize = [0.2, 0.4]
		self.list_sample = ["./sample/1.jpg"]
		self.list_label = ["./sample/1.txt"]
		self.backgrand = []

	def transform(self):
		dirList = os.listdir('./backgrand/')
		for i in range(len(dirList)):
			self.backgrand.append('./backgrand/'+str(dirList[i]))

	def IOU(self, labels_out, labels_inside, w, h):
		labels = np.concatenate((labels_out, labels_inside),axis = 0)
		n = labels.shape[0]
		x1 = (w * (labels[:,1] - labels[:,3]/2))  #左下坐标x
		y1 = (h * (labels[:,2] - labels[:,4]/2))  #左下坐标y
		x2 = (w * (labels[:,1] + labels[:,3]/2))  #右上坐标x
		y2 = (h * (labels[:,2] + labels[:,4]/2))  #右上坐标y
		xx1 = np.maximum(x1[0], x1[1:n])
		yy1 = np.maximum(y1[0], y1[1:n])
		xx2 = np.minimum(x2[0], x2[1:n])
		yy2 = np.minimum(y2[0], y2[1:n])

		w_1 = np.maximum(0.0, xx2 - xx1 + 1)
		h_1 = np.maximum(0.0, yy2 - yy1 + 1)
		# print(w * h)
		ovr = w_1 * h_1 / (labels_inside[:,3] * w * labels_inside[:,4] * h)   #IOU
		check = labels_inside[np.argwhere(ovr > 0.8)].reshape(-1,5)
		check[:,1] = np.max( (check[:,1] - (labels_out[:,1] - labels_out[:,3]/2)) / labels_out[:,3], 0)
		check[:,2] = np.max( (check[:,2] - (labels_out[:,2] - labels_out[:,4]/2)) / labels_out[:,4], 0)
		check[:,3] = np.max(check[:,3] / labels_out[:,3], 0)
		check[:,4] = np.max(check[:,4] / labels_out[:,4], 0)

		return check

	def paste(self, item, check, bg, Mask):
		h_bg, w_bg, _ = bg.shape
		h, w, _ = item.shape
		size_rate =  random.uniform(self.random_resize[0], self.random_resize[1])
		if w >= h:
			new_w = int(w_bg * size_rate)
			new_h = int(h / w * w_bg * size_rate)
		else:
			new_w = int(w / h * h_bg * size_rate)
			new_h = int(h_bg * size_rate)
		item = cv2.resize(item, (new_w, new_h))
		edge = [1 - new_w/w_bg, 1 - new_h/h_bg]
		x = random.randint(0, int(edge[0] * w_bg)) - 1
		y = random.randint(0, int(edge[1] * h_bg)) - 1
		if x <=0:
			x = 0
		if y <=0:
			y = 0
		label = None
		if not Mask[y:y+new_h, x:x+new_w, :].any():
			bg[y:y+new_h, x:x+new_w,:] = np.where(item > 1, item, bg[y:y+new_h, x:x+new_w,:])
			Mask[y:y+new_h, x:x+new_w,:] = 1
			label = np.array([0, (x + new_w/2)/w_bg, (y + new_h / 2)/h_bg, new_w/w_bg, new_h/h_bg])
			check[:,1] = np.max((x + check[:,1] * new_w)/w_bg, 0)
			check[:,2] = np.max((y + check[:,2] * new_h)/h_bg, 0)
			check[:,3] = np.max(check[:,3] * new_w / w_bg , 0)
			check[:,4] = np.max(check[:,4] * new_h / h_bg , 0)
			label = np.concatenate((label[np.newaxis,:], check), axis = 0)

		return bg , Mask, label

	def getitem(self):# 获取对象
		list_item = []    #
		list_inside = []
		#遍历样本
		for i in range(len(self.list_sample)):
			sample_path = self.list_sample[i]   #样本路径
			label_path = self.list_label[i]     #标签路径
			pic = cv2.imread(sample_path)
			labels = np.loadtxt(label_path).reshape(-1, 5)

			labels_out = labels[(np.argwhere(labels[:,0] == 0)),:].reshape(-1, 5)   #标签为1的类别
			labels_in = labels[(np.argwhere(labels[:,0] == 1)),:].reshape(-1, 5)    #标签为0的类别
			# print(labels_outer)
			# print(labels_inside)
			for i in range(labels_out.shape[0]):
				h, w, _ = pic.shape     #图片的长宽
				check = self.IOU(labels_out[i:i+1], labels_in, w, h)   #选择最佳的左上和右下坐标
				x1 = int(w * (labels_out[i, 1] - labels_out[i, 3]/2)) #左下坐标x
				y1 = int(h * (labels_out[i, 2] - labels_out[i, 4]/2)) #左下坐标y
				x2 = int(w * (labels_out[i, 1] + labels_out[i, 3]/2)) #右上坐标x
				y2 = int(h * (labels_out[i, 2] + labels_out[i, 4]/2)) #右上坐标y
				item = pic[y1:y2,x1:x2,:]    #截取目标区域
				list_item.append(item)      #目标的图片
				list_inside.append(check)   #框的坐标

		return list_item, list_inside

	def filling(self, bg):
		list_item, list_inside = self.getitem() # 目标
		h,w,c = bg.shape
		Mask = np.zeros((h,w,1))
		j = 0
		for k in range(len(list_item)):
			item = list_item[k] # list
			check = list_inside[k] # list
			for i in range(self.sample_item):
				mask = Mask
				c = check.copy()
				bg , Mask, label = self.paste(item, c, bg, mask)   #将目标粘贴到背景下
				if j == 0 and i == 0:
					new_label = label
				elif isinstance(label,np.ndarray):
					new_label = np.concatenate((new_label, label), axis=0)
			j += 1
		return bg, new_label

	def superimpose(self): # 叠加
		j = 1
		self.transform()
		print(self.backgrand)
		for bg_path in self.backgrand:
			for i in range(self.number_photo):
				bg = cv2.imread(bg_path)
				pic, label = self.filling(bg)
				cv2.imwrite("./data/" + str(j) + ".png",pic)

				label[:][0][0] = int(label[:][0][0])
				np.savetxt("./data/" + str(j) + ".txt", label, fmt='%d %f %f %f %f',delimiter=' ')
				j += 1

if __name__ == '__main__':
	s = Superimpose()
	s.superimpose()

	# img = cv2.imread("./data/1.png")# "./sample/1.jpg" "./data/1.png"
	# h,w,_ = img.shape
	# labels = np.loadtxt("./data/1.txt").reshape(-1, 5) # "./sample/1.txt"
	# x1 = w * (labels[:, 1] - labels[:, 3]/2)
	# y1 = h * (labels[:, 2] - labels[:, 4]/2)
	# x2 = w * (labels[:, 1] + labels[:, 3]/2)
	# y2 = h * (labels[:, 2] + labels[:, 4]/2)

	# for i in range(labels.shape[0]):
	# 	cv2.rectangle(img,(int(x1[i]),int(y1[i])),(int(x2[i]),int(y2[i])),(0,255,0),3)
	# cv2.imwrite("./1.png",img)

