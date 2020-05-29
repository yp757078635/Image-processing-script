'''
Python批量实现图像水平、垂直翻转
函数功能：扩大数据量
'''
import PIL.Image as img
import os

path_old = r"radish"
path_new = r"new_radish"
filelist = os.listdir(path_old)

for subdir in filelist:
    sub_dir=path_old+'/'+subdir
    im=img.open(sub_dir)
    im.save(sub_dir)
    ng1=im.transpose(img.ROTATE_180)
    ng1.save(path_old+'/'+"(1)"+subdir)
    ng2=im.transpose(img.FLIP_LEFT_RIGHT)
    ng2.save(path_old + '/' + "(2)"+subdir)
    ng3=im.transpose(img.FLIP_TOP_BOTTOM)
    ng3.save(path_old + '/'+"(3)"+subdir)
print("Done")





# im.show()
# ng.show()









