import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.linalg import eig  # 添加必要的库

# Trigger 定义
# 实验开始  实验结束  Block开始  Block结束  Trial开始  Trial结束  左手想象  右手想象  双脚想象  测试集特有(想象开始)
#   250      251      242       243        240       241       201     202      203         249

# 定义常量
ID_NUM = 5
BLOCK_NUM = 25
K = 6  # 投影方向W为64xK的矩阵
# L = 64 # 通道数
INTERPRET_TASK = {'1': 'left', '2': 'right', '3': 'feet'}


class Train():
    # @brief: 初始化函数, 初始化数据路径、数据字典、协方差矩阵字典
    def __init__(self):
        self.root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../MI_data_training/'))
        
        # 数据字典, key为受试者ID, value为一个字典, key为左手、右手、双脚 三类任务数据, 每个数据为一个64*0的矩阵
        self.data = {id: {'left': np.zeros((64, 0)), 'right': np.zeros((64, 0)), 'feet': np.zeros((64, 0))} for id in range(1, 6)}
        # print("data:", self.data)

        # 三类样本的协方差矩阵字典, key为受试者ID, value为一个字典, key为左手、右手、双脚三个数据, 每个数据为一个64*64的矩阵
        self.cov_matrix_task_all = {id: {'left': np.zeros((64, 64)), 'right': np.zeros((64, 64)), 'feet': np.zeros((64, 64))} for id in range(1, 6)}

        # Σ_x1x1、Σ_x2x2, 两类样本的协方差矩阵, 1和2分别指代任务1和任务2
        self.cov_matrix_task = {'1': np.zeros((64, 64)), '2': np.zeros((64, 64))}
        
        # Σ_xx, 总体协方差矩阵
        self.cov_matrix_all = np.zeros((64, 64))

        # Σ_xx^(1/2), 总体数据的白化矩阵
        self.whitening_matrix_all = np.zeros((64, 64))

        # Σ_x1'x1'、Σ_x2'x2', 两类样本白化后的协方差矩阵
        self.cov_matrix_task_whitened = {'1': np.zeros((64, 64)), '2': np.zeros((64, 64))}

        # W', 投影矩阵
        self.W_prime = np.zeros((64, 64))

        # W, 最优空间滤波矩阵
        self.W = np.zeros((64, 64))

        # Y1', Y2', 样本经过CSP空域滤波器的投影
        self.Y1_prime = self.Y2_prime = np.zeros((K, 0))




    # @brief: 主函数
    def run(self):
        self.preprocess_data('r') # 从pkl文件读取预处理后的数据和三类样本的协方差矩阵
        for id in range(1, ID_NUM + 1): # 几位受试者分别进行
            for tasks in ['12', '13', '23']: # 12、13、23三个CSP的训练分别进行
                x1 = tasks[0]
                x2 = tasks[1]
                # 1.载入两类样本的协方差矩阵: Σ_x1x1、Σ_x2x2
                self.cov_matrix_task['1'] = self.cov_matrix_task_all[id][INTERPRET_TASK[x1]]
                self.cov_matrix_task['2'] = self.cov_matrix_task_all[id][INTERPRET_TASK[x2]]
                # 2.计算总体的样本协方差矩阵: Σ_xx
                self.cov_matrix_all = self.cov_matrix_task['1'] + self.cov_matrix_task['2']
                # 3.计算总体数据的白化矩阵: Σ_xx^(1/2)
                self.whitening_matrix_all = self.compute_whitening_matrix(self.cov_matrix_all)
                print("whitening matrix for id:", id, " task:", tasks, ":\n", self.whitening_matrix_all)
                # 4.计算两类样本白化后的协方差矩阵: Σ_x1'x1'、Σ_x2'x2'
                self.cov_matrix_task_whitened['1'] = self.whitening_matrix_all @ self.cov_matrix_task['1'] @ self.whitening_matrix_all.T
                self.cov_matrix_task_whitened['2'] = self.whitening_matrix_all @ self.cov_matrix_task['2'] @ self.whitening_matrix_all.T
                # 5.计算广义特征值分解，获得投影矩阵 W'
                eigvals, eigvecs = eig(self.cov_matrix_task_whitened['1'], self.cov_matrix_task_whitened['2'])
                idx = np.argsort(eigvals)[::-1] # 将特征值和特征向量按特征值从大到小排序
                eigvecs = eigvecs[:, idx]
                self.W_prime = np.hstack((eigvecs[:, :K//2], eigvecs[:, -K//2:])).real # 提取前K/2列和后K/2列拼接
                # print("W_prime for id:", id, " task:", tasks, ":\n", self.W_prime)
                # print("W_prime shape:", self.W_prime.shape)
                # 6.计算最优空间滤波矩阵W
                self.W = self.whitening_matrix_all.T @ self.W_prime
                # 7.采用所有训练样本通过共空间滤波矩阵W, 计算投影Y'
                self.Y1_prime =  self.W.T @ self.data[id][INTERPRET_TASK[x1]] # 64*K @ K*N = 64*N
                self.Y2_prime =  self.W.T @ self.data[id][INTERPRET_TASK[x2]]
                # 8.计算两类样本Y', 每行的自相关系数，并训练分类器
                
                
                
        return


    # @brief: 读取pkl文件内数据
    # @param: id: 从1到5,代表S1到S5, 5位受试者
    # @param: block: 从1到25, 代表第1到第25个block数据
    # @param: channel: 从1到65, 代表第1到第65个通道数据
    # @return: 返回数据
    def get_data(self, id, block, channel=None):
        data = joblib.load(self.root_dir + '/S' + str(id) + '/block_' + str(block) + '.pkl')['data']
        if channel is not None:
            return data[channel - 1]
        else:
            return data


    # @brief: 数据概览
    # @param: data: 数据
    # @param: downsampling: 降采样, None表示不降采样, 250表示结果为原数据的1/250
    def test_overview(self, data, downsampling=None):
        print("original shape:",data.shape)
        if data.ndim == 1: data = data[::downsampling]
        if data.ndim == 2: data = data[:,::downsampling]
        # data = data[~np.isin(data, [0, 201, 202, 203])] # 去除这几个元素
        print(data)
        if (downsampling is not None): print("down sampled shape:", data.shape)
        return data


    # @brief: 计算白化矩阵
    def compute_whitening_matrix(self, cov_matrix):
        # 在协方差矩阵上添加一个小的正则化项
        cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
        eigvals, eigvecs = np.linalg.eigh(cov_matrix)  # 特征分解, (eigh适用于对称矩阵)
        eigvals_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals)) # 计算 Λ^(-1/2)
        # 构建白化矩阵 Σ_xx^(-1/2)
        whitening_matrix = (eigvals_inv_sqrt @ eigvecs).T # @为矩阵乘法
        return whitening_matrix
    

    # @brief: 预处理数据
    #         w模式读取原始数据并分类, 保存三类样本的协方差矩阵至pkl文件
    #         r模式直接读取之前预处理的结果, 从pkl文件读取三类样本的协方差矩阵
    def preprocess_data(self,operation):
        if operation == 'w':  # 保存三类样本的协方差矩阵至pkl文件
            # 读取数据并分类, 存入self.data字典
            for id in range(1, ID_NUM + 1):
                for block in range(1, BLOCK_NUM + 1):
                    data = self.get_data(id=id, block=block)
                    # data = data[:, ~np.isin(data[64, :], [0, 242, 243])] # 去除65号通道这几个元素所在的列
                    # 将数据按照trigger分类
                    self.data[id]['left'] = np.hstack((self.data[id]['left'], data[:64, data[64, :] == 201]))
                    self.data[id]['right'] = np.hstack((self.data[id]['right'], data[:64, data[64, :] == 202]))
                    self.data[id]['feet'] = np.hstack((self.data[id]['feet'], data[:64, data[64, :] == 203]))
            # 将数据转换为三类样本的协方差矩阵
            for id in range(1, ID_NUM + 1):
                for task in ['left', 'right', 'feet']:
                    if self.data[id][task].size > 0:
                        # 保险起见, 手动去中心化, 不加这步直接调cov发现会算出负数的特征值
                        self.data[id][task] -= np.mean(self.data[id][task], axis=1, keepdims=True)
                        self.cov_matrix_task_all[id][task] = np.cov(self.data[id][task])
                    else:
                        self.cov_matrix_task_all[id][task] = np.zeros((64, 64))
            # print("cov_matrix_task_all:", self.cov_matrix_task_all)
            # 保存预处理后的数据和三类样本的协方差矩阵至pkl文件
            with open(os.path.join(os.path.dirname(__file__), 'data.pkl'), 'wb') as f:
                joblib.dump(self.data, f)
            with open(os.path.join(os.path.dirname(__file__), 'cov_matrix_task_all.pkl'), 'wb') as f:
                joblib.dump(self.cov_matrix_task_all, f)
        elif operation == 'r': # 从pkl文件读取预处理后的数据和三类样本的协方差矩阵
            with open(os.path.join(os.path.dirname(__file__), 'data.pkl'), 'rb') as f:
                self.data = joblib.load(f)
            with open(os.path.join(os.path.dirname(__file__), 'cov_matrix_task_all.pkl'), 'rb') as f:
                self.cov_matrix_task_all = joblib.load(f)





if __name__ == '__main__':
    train = Train()
    train.run()