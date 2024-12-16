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


    # @brief: 主函数
    def run(self):
        # data = self.get_data(1,1)
        # data = self.test_overview(data,250)
        # trigger = self.get_data(1,1,65) # 第65通道为trigger
        # trigger = self.test_overview(trigger)
        # print("trigger:", trigger)

        # 读取数据并分类
        self.data_sort()
        # print(self.data[1]['left'].shape)
        # print(self.data[1]['left'])
        # print(self.data[1]['right'].shape)
        # print(self.data[1]['right'])
        # print(self.data[1]['feet'].shape)
        # print(self.data[1]['feet'])

        # 将数据转换为三类样本的协方差矩阵
        self.data_to_cov_matrix()
        # print(self.cov_matrix_task[1]['left'].shape)
        # print(self.cov_matrix_task[1]['left'])

        # 保存三类样本的协方差矩阵至pkl文件
        with open(os.path.join(os.path.dirname(__file__), 'cov_matrix_task.pkl'), 'wb') as f:
            joblib.dump(self.cov_matrix_task, f)

        # 从pkl文件读取三类样本的协方差矩阵
        # with open(os.path.join(os.path.dirname(__file__), 'cov_matrix_task.pkl'), 'rb') as f:
        #     self.cov_matrix_task = joblib.load(f)
        # print("cov_matrix_task:", self.cov_matrix_task)

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
                    self.cov_matrix_task[id][task] = np.cov(self.data[id][task])
                else:
                    self.cov_matrix_task[id][task] = np.zeros((64, 64))






if __name__ == '__main__':
    train = Train()
    train.run()