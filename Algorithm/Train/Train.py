import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import butter, filtfilt
from scipy.linalg import eig  # 添加必要的库

# Trigger 定义
# 实验开始  实验结束  Block开始  Block结束  Trial开始  Trial结束  左手想象  右手想象  双脚想象  测试集特有(想象开始)
#   250      251      242       243        240       241       201     202      203         249

# 定义常量
ID_NUM = 5
BLOCK_NUM = 25
L = 59  # 脑电信号通道数
K = 6  # 投影方向W为 LxK 的矩阵
INTERPRET_TASK = {'1': 'left', '2': 'right', '3': 'feet'}

# 定义多个频率带
frequency_bands = [(1, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48)]

# 定义采样频率
fs = 250  # 根据实际情况设置采样频率


class Train():
    # @brief: 初始化函数, 初始化数据路径、数据字典、协方差矩阵字典
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MI_data_training/'))

        # 数据字典, key为受试者ID, value为一个字典, key为左手、右手、双脚 三类任务数据, 每个数据为一个列表，列表中的每个元素为一个LxN的矩阵
        self.data = {id: {'left': [], 'right': [], 'feet': []} for id in range(1, 6)}
        # print("data:", self.data)

        self.data_index = 0 # 用于按trail获取数据的临时索引

        # 有效的组数, left+right+feet 算作一组, 多余的数据舍弃
        self.valid_group_num = {id: 0 for id in range(1, 6)}

        # 三类样本的协方差矩阵字典, key为受试者ID, value为一个字典, key为左手、右手、双脚三个数据, 每个数据为一个列表，列表中的每个元素为一个LxL的矩阵
        self.cov_matrix_task_all = {id: {'left': None, 'right': None, 'feet': None} for id in range(1, 6)}

        # Σ_x1x1、Σ_x2x2, 两类样本的协方差矩阵, 1和2分别指代任务1和任务2
        self.cov_matrix_task = {'1': np.zeros((L, L)), '2': np.zeros((L, L))}

        # Σ_xx, 总体协方差矩阵
        self.cov_matrix_all = np.zeros((L, L))

        # Σ_xx^(1/2), 总体数据的白化矩阵
        self.whitening_matrix_all = np.zeros((L, L))

        # Σ_x1'x1'、Σ_x2'x2', 两类样本白化后的协方差矩阵
        self.cov_matrix_task_whitened = {'1': np.zeros((L, L)), '2': np.zeros((L, L))}

        # W', 投影矩阵
        self.W_prime = np.zeros((L, L))

        # W, 最优空间滤波矩阵
        self.W = np.zeros((L, L))

        self.scores = []

    # @brief: 主函数
    def run(self):
        self.preprocess_data('w')  # 从pkl文件读取预处理后的数据和三类样本的协方差矩阵
        for id in range(1, 6):
            # 初始化每个任务的特征列表
            features_task = {'12': {'feature_1': [], 'feature_2': []},
                             '13': {'feature_1': [], 'feature_2': []},
                             '23': {'feature_1': [], 'feature_2': []}}
            
            for band_idx, band in enumerate(frequency_bands):
                # 进行带通滤波
                filtered_data = {
                    task: [self.bandpass_filter(trial, band[0], band[1]) for trial in self.data[id][task]]
                    for task in ['left', 'right', 'feet']
                }
                for task in ['12', '13', '23']:
                    x1 = task[0]
                    x2 = task[1]
                    # 1.载入两类样本的协方差矩阵: Σ_x1x1、Σ_x2x2
                    # print("id:", id, "block:", block)
                    self.cov_matrix_task['1'] = self.cov_matrix_task_all[id][INTERPRET_TASK[x1]]
                    self.cov_matrix_task['2'] = self.cov_matrix_task_all[id][INTERPRET_TASK[x2]]
                    # 2.计算总体的样本协方差矩阵: Σ_xx
                    self.cov_matrix_all = self.cov_matrix_task['1'] + self.cov_matrix_task['2']
                    # 3.计算总体数据的白化矩阵: Σ_xx^(1/2)
                    self.whitening_matrix_all = self.compute_whitening_matrix(self.cov_matrix_all)
                    # print("whitening matrix for id:", id, " task:", task, ":\n", self.whitening_matrix_all)
                    # 4.计算两类样本白化后的协方差矩阵: Σ_x1'x1'、Σ_x2'x2'
                    self.cov_matrix_task_whitened['1'] = self.whitening_matrix_all @ self.cov_matrix_task[
                        '1'] @ self.whitening_matrix_all.T
                    self.cov_matrix_task_whitened['2'] = self.whitening_matrix_all @ self.cov_matrix_task[
                        '2'] @ self.whitening_matrix_all.T
                    # 5.计算广义特征值分解，获得投影矩阵 W'
                    eigvals, eigvecs = eig(self.cov_matrix_task_whitened['1'], self.cov_matrix_task_whitened['2'])
                    idx = np.argsort(eigvals)[::-1]  # 将特征值和特征向量按特征值从大到小排序
                    eigvecs = eigvecs[:, idx]
                    self.W_prime = np.hstack((eigvecs[:, :K // 2], eigvecs[:, -K // 2:])).real  # 提取前K/2列和后K/2列拼接
                    # 6.计算最优空间滤波矩阵W
                    self.W = self.whitening_matrix_all.T @ self.W_prime  # LxL @ LxK = LxK
                    # 修改保存模型的文件名，包含频率带索引
                    with open(os.path.join(os.path.dirname(__file__), '../model/S' + str(id) + f'/csp_{task}_band{band_idx}.pkl'),
                              'wb') as f:
                        joblib.dump(self.W.T, f)
                    
                    for block in range(0, self.valid_group_num[id]):  # 从0到self.valid_group_num[id]-1
                        # 7.采用所有训练样本通过共空间滤波矩阵W, 计算投影Y'
                        Y1_prime = self.W.T @ filtered_data[INTERPRET_TASK[x1]][block]  # KxL @ LxN = KxN
                        Y2_prime = self.W.T @ filtered_data[INTERPRET_TASK[x2]][block]
                        # 8.计算两类样本Y', 每行的自相关系数
                        features_task[task]['feature_1'].append(np.diag(np.cov(Y1_prime)))  # 长度为K
                        features_task[task]['feature_2'].append(np.diag(np.cov(Y2_prime)))  # 长度为K
                        # print("feature 1:", len(features_task[task]['feature_1']), "feature 2:", len(features_task[task]['feature_2']))
            
            # 训练LDA模型使用所有频率带的特征
            for task in ['12', '13', '23']:
                feature_1 = np.hstack(features_task[task]['feature_1'])  # 6 * num_bands = 18
                feature_2 = np.hstack(features_task[task]['feature_2'])  # 6 * num_bands = 18
                # 重新划分为二维数组，每个band的特征相邻
                feature_1 = feature_1.reshape(-1, K * len(frequency_bands))
                feature_2 = feature_2.reshape(-1, K * len(frequency_bands))
                # 9.训练分类器
                self.train_lda(id, task, feature_1, feature_2)

        print("mean scores:", np.mean(self.scores))
        return

    def train_lda(self, id, task, feature_1, feature_2):
        lda = LDA()  # 训练LDA模型
        X = np.vstack((feature_1, feature_2))
        y = np.hstack((np.zeros(len(feature_1)), np.ones(len(feature_2))))
        lda.fit(X, y)
        # print("lda score:", lda.score(X, y))
        # 保存训练模型
        with open(os.path.join(os.path.dirname(__file__), '../model/S' + str(id) + '/lda_' + task + '.pkl'), 'wb') as f:
            joblib.dump(lda, f)
        self.scores.append(lda.score(X, y))


    # @brief: 读取pkl文件内数据
    # @param: id: 从1到5,代表S1到S5, 5位受试者
    # @param: block: 从1到25, 代表第1到第25个block数据
    # @param: channel: 从1到65, 代表第1到第65个通道数据
    # @return: 返回数据
    def get_data(self, id, block, channel=None):
        data = joblib.load(self.root_dir + '/S' + str(id) + '/block_' + str(block) + '.pkl')['data']
        cutoff_index = self.data_index
        # 读取数据, 直到遇到241为止
        while data[64, cutoff_index] != 241:
            cutoff_index += 1
            if cutoff_index == data.shape[1]:
                return np.array([])
        data = data[:, self.data_index:cutoff_index]
        # print("data shape:", data.shape)
        # print("data:", data)
        self.data_index = cutoff_index + 1
        if channel is not None:
            return data[channel - 1]
        else:
            return data

    # @brief: 数据概览
    # @param: data: 数据
    # @param: downsampling: 降采样, None表示不降采样, 250表示结果为原数据的1/250
    def test_overview(self, data, downsampling=None):
        print("original shape:", data.shape)
        if data.ndim == 1: data = data[::downsampling]
        if data.ndim == 2: data = data[:, ::downsampling]
        # data = data[~np.isin(data, [0, 201, 202, 203])] # 去除这几个元素
        print(data)
        if (downsampling is not None): print("down sampled shape:", data.shape)
        return data

    # @brief: 计算白化矩阵
    def compute_whitening_matrix(self, cov_matrix):
        # 在协方差矩阵上添加一个小的正则化项
        cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # 特征分解, (eigh适用于对称矩阵)
        eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))  # 计算 Λ^(-1/2)
        # 构建白化矩阵 Σ_xx^(-1/2)
        whitening_matrix = (eigvals_inv_sqrt @ eigvecs).T  # @为矩阵乘法
        return whitening_matrix

    # @brief: 预处理数据
    #         w模式读取原始数据并分类, 保存三类样本的协方差矩阵至pkl文件
    #         r模式直接读取之前预处理的结果, 从pkl文件读取三类样本的协方差矩阵
    def preprocess_data(self, operation):
        if operation == 'w':  # 保存三类样本的协方差矩阵至pkl文件
            task_map = {201: 'left', 202: 'right', 203: 'feet'}
            # 读取数据并分类, 存入self.data字典
            for id in range(1, ID_NUM + 1):
                for block in range(1, BLOCK_NUM + 1):
                    # data = data[:, ~np.isin(data[L, :], [0, 242, 243])] # 去除65号通道这几个元素所在的列
                    self.data_index = 0
                    while True:
                        data = self.get_data(id=id, block=block)
                        if data.size == 0:
                            break
                        # 将数据按照trigger分类
                        # print("block:", block, "data:", data)
                        for trigger, task in task_map.items():
                            mask = data[64, :] == trigger
                            # print("trigger:", trigger, "mask:", mask)
                            if mask.any():
                                temp = data[:, mask[:data.shape[1]]]
                                assert np.all(temp[64, :] == trigger)
                                temp[:L, :] -= np.mean(temp[:L, :], axis=1, keepdims=True)
                                self.data[id][task].append(temp[:L, :])
                                
                # print("id:", id, "left num:", len(self.data[id]['left']), "right num:", len(self.data[id]['right']), "feet num:", len(self.data[id]['feet']))
                self.valid_group_num[id] = min(len(self.data[id]['left']), len(self.data[id]['right']),len(self.data[id]['feet']))
                for task in ['left', 'right', 'feet']:
                    self.data[id][task] = self.data[id][task][:self.valid_group_num[id]]
                # print("id:", id, "left num:", len(self.data[id]['left']), "right num:", len(self.data[id]['right']), "feet num:", len(self.data[id]['feet']))
            # print(self.data[1]['left'])
            # print("valid group num:", self.valid_group_num)
            # 将数据转换为三类样本的协方差矩阵
            for id in range(1, ID_NUM + 1):
                for task in ['left', 'right', 'feet']:
                    data = np.hstack(self.data[id][task])
                    # print("data shape:", data.shape)
                    data -= np.mean(data, axis=1, keepdims=True)  # 去均值
                    self.cov_matrix_task_all[id][task] = np.cov(data)
            # print("cov_matrix_task_all:", self.cov_matrix_task_all)
            # 保存预处理后的数据和三类样本的协方差矩阵至pkl文件
            with open(os.path.join(os.path.dirname(__file__), 'data.pkl'), 'wb') as f:
                joblib.dump(self.data, f)
            with open(os.path.join(os.path.dirname(__file__), 'cov_matrix_task_all.pkl'), 'wb') as f:
                joblib.dump(self.cov_matrix_task_all, f)
            with open(os.path.join(os.path.dirname(__file__), 'valid_group_num.pkl'), 'wb') as f:
                joblib.dump(self.valid_group_num, f)
        elif operation == 'r':  # 从pkl文件读取预处理后的数据和三类样本的协方差矩阵
            with open(os.path.join(os.path.dirname(__file__), 'data.pkl'), 'rb') as f:
                self.data = joblib.load(f)
            with open(os.path.join(os.path.dirname(__file__), 'cov_matrix_task_all.pkl'), 'rb') as f:
                self.cov_matrix_task_all = joblib.load(f)
            with open(os.path.join(os.path.dirname(__file__), 'valid_group_num.pkl'), 'rb') as f:
                self.valid_group_num = joblib.load(f)

    # 添加带通滤波函数
    def bandpass_filter(self, data, low, high):
        Wn = [low / (fs / 2), high / (fs / 2)]  # 规范化频率
        b, a = butter(N=4, Wn=Wn, btype='band')
        return filtfilt(b, a, data)


if __name__ == '__main__':
    train = Train()
    train.run()