import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
from scipy import signal
from collections import namedtuple

# Trigger 定义
# 实验开始	实验结束	 Block开始	Block结束	Trial开始	Trial结束	左手想象	右手想象	双脚想象	测试集特有(想象开始)
#  250        251     242         243         240         241         201     202         203         249

class Train():
    # @brief: 初始化函数, 初始化数据路径、数据字典、协方差矩阵字典
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MI_data_training/'))
        
        # 数据字典，key为受试者ID，value为一个字典，key为左手、右手、双脚三个数据，每个数据为一个64*0的矩阵
        self.data = {id: {'left': np.zeros((64, 0)), 'right': np.zeros((64, 0)), 'feet': np.zeros((64, 0))} for id in range(1, 6)}
        # print("data:", self.data)
        # print(type(self.data))

        # 协方差矩阵字典，key为受试者ID，value为一个字典，key为左手、右手、双脚三个数据，每个数据为一个64*64的矩阵
        self.cov_matrix_task = {id: {'left': np.zeros((64, 64)), 'right': np.zeros((64, 64)), 'feet': np.zeros((64, 64))} for id in range(1, 6)}

        # 总体的样本协方差矩阵，key为受试者ID，value为一个字典,key为12(左右手)，13(左手双脚)，23(右手双脚)三个数据，每个数据为一个64*64的矩阵
        self.cov_matrix_all ={id: {'12': np.zeros((64, 64)), '13': np.zeros((64, 64)), '23': np.zeros((64, 64))} for id in range(1, 6)}

        # 总体数据的白化矩阵，key为受试者ID，value为一个字典,key为12(左右手)，13(左手双脚)，23(右手双脚)三个数据，每个数据为一个64*64的矩阵
        self.whitening_matrix_all = {id: {'12': np.zeros((64, 64)), '13': np.zeros((64, 64)), '23': np.zeros((64, 64))} for id in range(1, 6)}

        # 三类样本白化后协方差矩阵，key为受试者ID，value为一个字典,key12(左右手)，13(左手双脚)，23(右手双脚)三组数据，每组数据为两个64*64的矩阵
        self.cov_matrix_task_whitened = {id: {'12':[np.zeros((64, 64)), np.zeros((64, 64))], '13':[np.zeros((64, 64)), np.zeros((64, 64))], '23':[np.zeros((64, 64)), np.zeros((64, 64))]} for id in range(1, 6)}



    # @brief: 主函数
    def run(self):

        self.step_1_cov_matrix_task('r')

        # 分别计算总体的样本协方差矩阵，12，13，23的协方差矩阵
        for id in range(1, 6):
            self.cov_matrix_all[id]['12'] = self.cov_matrix_task[id]['left'] + self.cov_matrix_task[id]['right']
            self.cov_matrix_all[id]['13'] = self.cov_matrix_task[id]['left'] + self.cov_matrix_task[id]['feet']
            self.cov_matrix_all[id]['23'] = self.cov_matrix_task[id]['right'] + self.cov_matrix_task[id]['feet']
        # print("cov_matrix_all:", self.cov_matrix_all)
        # print(self.cov_matrix_all[5]['12'].shape)

        # 计算白化矩阵
        self.step_3_whitening_matrix_all('r')
        print("whitening_matrix_all:", self.whitening_matrix_all)

        # 计算三类样本白化后协方差矩阵
        self.step_4_cov_matrix_task_whitened('w')

        return




    # @brief: 读取pkl文件内数据
    # @param: id: 从1到5,代表S1到S5，5位受试者
    # @param: block: 从1到25，代表第1到第25个block数据
    # @param: channel: 从1到65，代表第1到第65个通道数据
    # @return: 返回数据
    def get_data(self, id, block, channel=None):
        data = joblib.load(self.root_dir + '/S' + str(id) + '/block_' + str(block) + '.pkl')['data']
        if channel is not None:
            return data[channel - 1]
        else:
            return data


    # @brief: 数据概览
    # @param: data: 数据
    # @param: downsampling: 降采样，None表示不降采样，250表示结果为原数据的1/250
    def test_overview(self, data, downsampling=None):
        print("original shape:",data.shape)
        if data.ndim == 1: data = data[::downsampling]
        if data.ndim == 2: data = data[:,::downsampling]
        # data = data[~np.isin(data, [0, 201, 202, 203])] # 去除这几个元素
        print(data)
        if (downsampling is not None): print("down sampled shape:", data.shape)
        return data
    

    # @brief: 读取数据并数据分类，存入self.data字典
    def data_sort(self):
        for id in range(1, 6): # (1,6)
            for block in range(1, 26): # (1,26)
                data = self.get_data(id=id, block=block)
                # data = data[:, ~np.isin(data[64, :], [0, 242, 243])] # 去除65号通道这几个元素所在的列
                # 将数据按照trigger分类
                self.data[id]['left'] = np.hstack((self.data[id]['left'], data[:64, data[64, :] == 201]))
                self.data[id]['right'] = np.hstack((self.data[id]['right'], data[:64, data[64, :] == 202]))
                self.data[id]['feet'] = np.hstack((self.data[id]['feet'], data[:64, data[64, :] == 203]))


    # @brief: 将数据转换为三类样本的协方差矩阵
    def data_to_cov_matrix(self):
        for id in range(1, 6):
            for task in ['left', 'right', 'feet']:
                if self.data[id][task].size > 0:
                    # 保险起见，手动去中心化，不加这步直接调cov发现会算出负数的特征值
                    self.data[id][task] -= np.mean(self.data[id][task], axis=1, keepdims=True)
                    self.cov_matrix_task[id][task] = np.cov(self.data[id][task])
                else:
                    self.cov_matrix_task[id][task] = np.zeros((64, 64))

    # @brief: 计算白化矩阵
    def compute_whitening_matrix(self, cov_matrix):
        # 在协方差矩阵上添加一个小的正则化项
        cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # 特征分解, (eigh适用于对称矩阵)
        eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals)) # 计算 Λ^(-1/2)
        # 构建白化矩阵 Σ_xx^(-1/2)
        whitening_matrix = (eigvals_inv_sqrt @ eigvecs).T # @为矩阵乘法
        return whitening_matrix
    
    def step_1_cov_matrix_task(self,operation):
        if operation == 'w':  # 保存三类样本的协方差矩阵至pkl文件
            self.data_sort() # 读取数据并分类
            self.data_to_cov_matrix() # 将数据转换为三类样本的协方差矩阵
            # print("cov_matrix_task:", self.cov_matrix_task)
            with open(os.path.join(os.path.dirname(__file__), 'cov_matrix_task.pkl'), 'wb') as f:
                joblib.dump(self.cov_matrix_task, f)
        elif operation == 'r': # 从pkl文件读取三类样本的协方差矩阵
            with open(os.path.join(os.path.dirname(__file__), 'cov_matrix_task.pkl'), 'rb') as f:
                self.cov_matrix_task = joblib.load(f)
    
    def step_3_whitening_matrix_all(self,operation):
        if operation == 'w': # 保存白化矩阵至pkl文件
            for id in range(1, 6): # 计算白化矩阵
                for tasks in ['12', '13', '23']:
                    self.whitening_matrix_all[id][tasks] = self.compute_whitening_matrix(self.cov_matrix_all[id][tasks])
            with open(os.path.join(os.path.dirname(__file__), 'whitening_matrix_all.pkl'), 'wb') as f:
                joblib.dump(self.whitening_matrix_all, f)
        elif operation == 'r': # 从pkl文件读取白化矩阵
            with open(os.path.join(os.path.dirname(__file__), 'whitening_matrix_all.pkl'), 'rb') as f:
                self.whitening_matrix_all = joblib.load(f)

    def step_4_cov_matrix_task_whitened(self,operation):
        if operation == 'w':
            for id in range(1, 6):
                for tasks in ['12', '13', '23']:
                    self.cov_matrix_task_whitened[id][tasks][0] = self.whitening_matrix_all[id][tasks] @ self.cov_matrix_task[id] # TODO



if __name__ == '__main__':
    train = Train()
    train.run()