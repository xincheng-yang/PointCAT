import open3d as o3d
import numpy as np
import os
import sys
import shutil

np.set_printoptions(suppress=True)  # 取消默认科学计数法，open3d无法读取科学计数法表示
# filedir = os.path.dirname(sys.argv[0])
# os.chdir(filedir)
# wdir = os.getcwd()
# wdir = wdir + '\\stanford_indoor3d'
# os.chdir(wdir)
# name = "Area_5_office_5"
# print('当前工作目录为：{}\n'.format(wdir))
# for parent, dirs, files in os.walk(wdir):
#     for file in files:
#         prefix = file.split('.')[0]
#         if prefix == name:
#             new_name = prefix + '.' + 'txt'
#             data = np.load(file)
#             b = np.array([1, 1, 1, 255, 255, 255])  # 每一列要除的数
#             np.savetxt(new_name, data[:, :6] / b)
#             print(new_ngt


# 读取点云并可视化
# name = "Area_5_office_5"
# pcd = o3d.io.read_point_cloud('./used_output/Area_5_office_5_pred_pointnet2.pcd')  # 原npy文件中的数据正好是按x y z r g b进行排列
pcd = o3d.io.read_point_cloud('stanford_indoor3d/Area_5_office_12.txt', format='xyzrgb')  # 原npy文件中的数据正好是按x y z r g b进行排列
print(pcd)
o3d.visualization.draw_geometries([pcd], width=2560, height=1440)
