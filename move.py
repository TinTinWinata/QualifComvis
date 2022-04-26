import shutil
import os
path = 'train/'
list_dir = os.listdir(path)

# print(list_dir)

for dir in list_dir:
    folder_name = dir[0:2]
    os.mkdir(path + folder_name)
    shutil.move(path + dir, path + folder_name)
    print('moved')
