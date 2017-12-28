import os

path = "F:\pywork\database\\footweight\\recognition\dongbo\\temp\V1.4.0.3\\NO3\\smalltest_128,59\\3\\"
path_exp = os.path.expanduser(path)
classes = [int(p) for p in os.listdir(path_exp)]
classes.sort()
# nrof_classes一个数据集下有多少个文件夹,就是说有多少个人,多少个类别
nrof_classes = len(classes)
count = 0
files = open("3test.txt", 'w')

count_u = 0
for i in range(nrof_classes):
    class_name = str(classes[i])
    count = count + 1
    count_u = count_u + 1
    facedir = os.path.join(path_exp, class_name)
    prefix1 =  path_exp+class_name + "\\"

    if os.path.isdir(facedir):
        images = os.listdir(facedir)

        image_paths = [(prefix1 + img + " " + class_name + "\n") for img in images]

        files.writelines(image_paths)

files.close()
