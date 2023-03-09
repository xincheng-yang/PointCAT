import tqdm
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import json
import warnings
import numpy as np
from torch.utils.data import Dataset
from model.pointcat_part import Vertax_elite
import torch.nn as nn
from torch.autograd import Variable
import copy

warnings.filterwarnings('ignore')
matplotlib.use("Agg")


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class PartNormalDataset(Dataset):
    def __init__(self, root='./data/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500,
                 split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints  # 采样点数
        self.root = root  # 文件根路径
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')  # 类别和文件夹名字对应的路径
        self.cat = {}
        self.normal_channel = normal_channel  # 是否使用rgb信息

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in
                    self.cat.items()}  # {'Airplane': '02691156', 'Bag': '02773838', 'Cap': '02954340', 'Car': '02958343', 'Chair': '03001627', 'Earphone': '03261776', 'Guitar': '03467517', 'Knife': '03624134', 'Lamp': '03636649', 'Laptop': '03642806', 'Motorbike': '03790512', 'Mug': '03797390', 'Pistol': '03948459', 'Rocket': '04099429', 'Skateboard': '04225987', 'Table': '04379243'}
        self.classes_original = dict(zip(self.cat, range(
            len(self.cat))))  # {'Airplane': 0, 'Bag': 1, 'Cap': 2, 'Car': 3, 'Chair': 4, 'Earphone': 5, 'Guitar': 6, 'Knife': 7, 'Lamp': 8, 'Laptop': 9, 'Motorbike': 10, 'Mug': 11, 'Pistol': 12, 'Rocket': 13, 'Skateboard': 14, 'Table': 15}

        if not class_choice is None:  # 选择一些类别进行训练  好像没有使用这个功能
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}  # 读取分好类的文件夹jason文件 并将他们的名字放入列表中
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])  # '928c86eabc0be624c2bf2dcc31ba1713' 这是第一个值
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])  # # 拿到对应一个文件夹的路径 例如第一个文件夹02691156
            fns = sorted(os.listdir(dir_point))  # 根据路径拿到文件夹下的每个txt文件 放入列表中
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if
                       fn[0:-4] in train_ids]  # 判断文件夹中的txt文件是否在 训练txt中，如果是，那么fns中拿到的txt文件就是这个类别中所有txt文件中需要训练的文件，放入fns中
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                "第i次循环  fns中拿到的是第i个文件夹中符合训练的txt文件夹的名字"
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(
                    os.path.join(dir_point, token + '.txt'))  # 生成一个字典，将类别名字和训练的路径组合起来  作为一个大类中符合训练的数据
                # 上面的代码执行完之后，就实现了将所有需要训练或验证的数据放入了一个字典中，字典的键是该数据所属的类别，例如飞机。值是他对应数据的全部路径
                # {Airplane:[路径1，路径2........]}
        #####################################################################################################################################################
        self.datapath = []
        for item in self.cat:  # self.cat 是类别名称和文件夹对应的字典
            for fn in self.meta[item]:
                self.datapath.append((item, fn))  # 生成标签和点云路径的元组， 将self.met 中的字典转换成了一个元组

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]
        # self.classes  将类别的名称和索引对应起来  例如 飞机 <----> 0
        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        """
        shapenet 有16 个大类，然后每个大类有一些部件 ，例如飞机 'Airplane': [0, 1, 2, 3] 其中标签为0 1  2 3 的四个小类都属于飞机这个大类
        self.seg_classes 就是将大类和小类对应起来
        """
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])
        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

    def __getitem__(self, index):
        if index in self.cache:  # 初始slef.cache为一个空字典，这个的作用是用来存放取到的数据，并按照(point_set, cls, seg)放好 同时避免重复采样
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]  # 根据索引 拿到训练数据的路径self.datepath是一个元组（类名，路径）
            cat = self.datapath[index][0]  # 拿到类名
            cls = self.classes[cat]  # 将类名转换为索引
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)  # size 20488,7 读入这个txt文件，共20488个点，每个点xyz rgb +小类别的标签
            if not self.normal_channel:  # 判断是否使用rgb信息
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)  # 拿到小类别的标签
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])  # 做一个归一化
        choice = np.random.choice(len(seg), self.npoints, replace=True)  # 对一个类别中的数据进行随机采样 返回索引，允许重复采样
        # resample
        point_set = point_set[choice, :]  # 根据索引采样

        seg = seg[choice]

        return point_set, cls, seg  # pointset是点云数据，cls十六个大类别，seg是一个数据中，不同点对应的小类别

    def __len__(self):
        return len(self.datapath)


class Generate_txt_and_3d_img:
    def __init__(self, img_root, target_root, num_classes, testDataLoader, model_dict, color_map=None):
        self.img_root = img_root  # 点云数据路径
        self.target_root = target_root  # 生成txt标签和预测结果路径
        self.testDataLoader = testDataLoader
        self.num_classes = num_classes
        self.color_map = color_map
        self.heat_map = False  # 控制是否输出heatmap
        self.label_path_txt = os.path.join(self.target_root, 'label_txt')  # 存放label的txt文件
        self.make_dir(self.label_path_txt)

        # 拿到模型 并加载权重
        self.model_name = []
        self.model = []
        self.model_weight_path = []

        for k, v in model_dict.items():
            self.model_name.append(k)
            self.model.append(v[0])
            self.model_weight_path.append(v[1])

        # 加载权重
        self.load_cheackpoint_for_models(self.model_name, self.model, self.model_weight_path)

        # 创建文件夹
        self.all_pred_image_path = []  # 所有预测结果的路径列表
        self.all_pred_txt_path = []  # 所有预测txt的路径列表
        for n in self.model_name:
            self.make_dir(os.path.join(self.target_root, n + '_predict_txt'))
            self.make_dir(os.path.join(self.target_root, n + '_predict_image'))
            self.all_pred_txt_path.append(os.path.join(self.target_root, n + '_predict_txt'))
            self.all_pred_image_path.append(os.path.join(self.target_root, n + '_predict_image'))
        "将模型对应的预测txt结果和img结果生成出来，对应几个模型就在列表中添加几个元素"

        self.generate_predict_to_txt()  # 生成预测txt
        self.draw_3d_img()  # 画图

    def generate_predict_to_txt(self):

        for batch_id, (points, label, target) in tqdm.tqdm(enumerate(self.testDataLoader),
                                                           total=len(self.testDataLoader), smoothing=0.9):

            # 点云数据、整个图像的标签、每个点的标签、  没有归一化的点云数据（带标签）torch.Size([1, 7, 2048])
            points = points.transpose(2, 1)
            # print('1',target.shape) # 1 torch.Size([1, 2048])
            xyz_feature_point = points[:, :6, :]
            xyz = points[:, :3, :]
            norm_plt = points[:, 3:, :]
            xyz, label, target, norm_plt = Variable(xyz.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
            # xyz, label, target, norm_plt = xyz.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
            # 将标签保存为txt文件
            point_set_without_normal = np.asarray(
                torch.cat([points.permute(0, 2, 1), target[:, :, None]], dim=-1)).squeeze(0)  # 代标签 没有归一化的点云数据  的numpy形式
            np.savetxt(os.path.join(self.label_path_txt, f'{batch_id}_label.txt'), point_set_without_normal,
                       fmt='%.04f')  # 将其存储为txt文件
            " points  torch.Size([16, 2048, 6])  label torch.Size([16, 1])  target torch.Size([16, 2048])"

            assert len(self.model) == len(self.all_pred_txt_path), '路径与模型数量不匹配，请检查'

            for n, model, pred_path in zip(self.model_name, self.model, self.all_pred_txt_path):
                seg_pred = model(xyz, norm_plt, self.to_categorical(
                    label.squeeze(1), num_classes))
                seg_pred = seg_pred.cpu().data.numpy()
                # ===================================================================
                # seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c
                if self.heat_map:
                    out = np.asarray(np.sum(seg_pred, axis=2))
                    seg_pred = (out - np.min(out) / (np.max(out) - np.min(out)))
                else:
                    seg_pred = np.argmax(seg_pred, axis=-1)  # 获得网络的预测结果 b n c
                # ===================================================================
                seg_pred = np.concatenate([np.asarray(xyz_feature_point), seg_pred[:, None, :]],
                                          axis=1).transpose((0, 2, 1)).squeeze(0)  # 将点云与预测结果进行拼接，准备生成txt文件
                svae_path = os.path.join(pred_path, f'{n}_{batch_id}.txt')
                np.savetxt(svae_path, seg_pred, fmt='%.04f')

    def draw_3d_img(self):
        #   调用matpltlib 画3d图像

        each_label = os.listdir(self.label_path_txt)  # 所有标签txt路径
        self.label_path_3d_img = os.path.join(self.target_root, 'label_3d_img')
        self.make_dir(self.label_path_3d_img)

        assert len(self.all_pred_txt_path) == len(self.all_pred_image_path)

        for i, (pre_txt_path, save_img_path, name) in enumerate(
                zip(self.all_pred_txt_path, self.all_pred_image_path, self.model_name)):
            each_txt_path = os.listdir(pre_txt_path)  # 拿到txt文件的全部名字

            for idx, (txt, lab) in tqdm.tqdm(enumerate(zip(each_txt_path, each_label)), total=len(each_txt_path)):
                if i == 0:
                    self.draw_each_img(os.path.join(self.label_path_txt, lab), idx, heat_maps=False)
                self.draw_each_img(os.path.join(pre_txt_path, txt), idx, name=name, save_path=save_img_path,
                                   heat_maps=self.heat_map)

        print(f'所有预测图片已生成完毕，请前往：{self.all_pred_image_path} 查看')

    def draw_each_img(self, root, idx, name=None, skip=1, save_path=None, heat_maps=False):
        """root：每个txt文件的路径"""
        points = np.loadtxt(root)[:, :3]  # 点云的xyz坐标
        points_all = np.loadtxt(root)  # 点云的所有坐标
        points = self.pc_normalize(points)
        skip = skip  # Skip every n points

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        point_range = range(0, points.shape[0], skip)  # skip points to prevent crash
        x = points[point_range, 0]
        z = points[point_range, 1]
        y = points[point_range, 2]

        "根据传入的类别数 自定义生成染色板  标签 0对应 随机颜色1  标签1 对应随机颜色2"
        if self.color_map is not None:
            color_map = self.color_map
        else:
            color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, num_classes))}
        Label = points_all[point_range, -1]  # 拿到标签
        # 将标签传入前面的字典，找到对应的颜色 并放入列表

        Color = list(map(lambda x: color_map[x], Label))

        ax.scatter(x,  # x
                   y,  # y
                   z,  # z
                   c=Color,  # Color,  # height data for color
                   s=25,
                   marker=".")
        ax.axis('auto')  # {equal, scaled}
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('off')  # 设置坐标轴不可见
        ax.grid(False)  # 设置背景网格不可见
        ax.view_init(elev=0, azim=0)

        if save_path is None:
            plt.savefig(os.path.join(self.label_path_3d_img, f'{idx}_label_img.png'), dpi=400, bbox_inches='tight',
                        transparent=True)
        else:
            plt.savefig(os.path.join(save_path, f'{idx}_{name}_img.png'), dpi=400, bbox_inches='tight',
                        transparent=True)

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def make_dir(self, root):
        if os.path.exists(root):
            print(f'{root} 路径已存在 无需创建')
        else:
            os.mkdir(root)

    def to_categorical(self, y, num_classes):
        """ 1-hot encodes a tensor """
        new_y = torch.eye(num_classes)[y.cpu().data.numpy(), ]
        if y.is_cuda:
            return new_y.cuda(non_blocking=True)
        return new_y

    def load_cheackpoint_for_models(self, name, model, cheackpoints):

        assert cheackpoints is not None, '请填写权重文件'
        assert model is not None, '请实例化模型'

        for n, m, c in zip(name, model, cheackpoints):
            print(f'正在加载{n}的权重.....')
            weight_dict = torch.load(os.path.join(c, 'best_insiou_model.pth'))
            m.load_state_dict(weight_dict['model'])
            print(f'{n}权重加载完毕')


if __name__ == '__main__':
    img_root = './data/shapenetcore_partanno_segmentation_benchmark_v0_normal'
    target_root = './results/Vertax_elite_2023_best'

    num_classes = 16
    choice_dataset = 'ShapeNet'
    model = Vertax_elite().cuda()
    model = nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    model.eval()
    TEST_DATASET = PartNormalDataset(root=img_root, npoints=2048, split='test', normal_channel=True)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=1, shuffle=False, num_workers=0,
                                                 drop_last=True)
    color_map = {idx: i for idx, i in enumerate(np.linspace(0, 0.9, 50))}
    model_dict = {
        'PointCAT': [model, './checkpoints/Vertax_elite_2022_12_1'],
    }
    c = Generate_txt_and_3d_img(img_root, target_root, num_classes, testDataLoader, model_dict, color_map)
