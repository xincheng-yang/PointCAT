import open3d as o3d
import numpy as np
import os
import sys
import shutil

np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示
filedir = os.path.dirname(sys.argv[0])
os.chdir(filedir)
wdir = os.getcwd()
wdir = wdir + '\\stanford_indoor3d'
os.chdir(wdir)
print('当前工作目录为：{}\n'.format(wdir))
for parent, dirs, files in os.walk(wdir):
    for file in files:
        prefix = file.split('.')[0]
        new_name = prefix + '.' + 'txt'
        data = np.load(file)
        b = np.array([1, 1, 1, 255, 255, 255])  # 每一列要除的数
        np.savetxt(new_name, data[:, :6] / b)
        print(new_name)
