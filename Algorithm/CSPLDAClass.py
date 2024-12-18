import numpy as np
import os
import joblib
from sklearn.svm import SVC  # 添加必要的库
from scipy import signal


rootdir = os.path.dirname(os.path.abspath(__file__))
# 仅供测试所用
class CSPLDAClass:
    def __init__(self):
        self.csp_12 = [None] * 6
        self.svm_12 = [None] * 6
        self.csp_13 = [None] * 6
        self.svm_13 = [None] * 6
        self.csp_23 = [None] * 6
        self.svm_23 = [None] * 6
        self.getmodel()

    def fbcsp_transform(self, data, task, id):
        if task == '12':
            FBCSP = self.csp_12
        elif task == '13':
            FBCSP = self.csp_13
        else:
            FBCSP = self.csp_23

        # output = np.zeros((data.shape[0], data.shape[1], data.shape[2])) # 初始化滤波后的EEG数据数组
        # bandVue = np.array([[1, 8], [8, 16], [16, 24], [24, 32], [32, 40], [40, 48]]) # 定义频带范围
        # for i in range(6): # 对每个频带进行处理
        #     # 设计带通滤波器
        #     b, a = signal.butter(4, [2 * bandVue[i, 0] / 250, 2 * bandVue[i, 1] / 250], 'bandpass', analog=True)
        #     # 对每个试次进行滤波
        #     for trail in range(data.shape[0]):
        #         for ch in range(data.shape[1]): # 对每个通道进行E滤波
        #             output[trail, ch, :] = signal.filtfilt(b, a, data[trail, ch, :])

        Y_prime = FBCSP[id - 1] @ data
        feature = np.diag(np.cov(Y_prime))
        return feature

    def getmodel(self):
        # 加载训练模型
        for id in range(1,6): 
            model_path = rootdir + '/model/S' + str(id) + '/'
            self.csp_12[id-1] = joblib.load(model_path + 'csp_12.pkl')
            self.svm_12[id-1] = joblib.load(model_path + 'svm_12.pkl')
            self.csp_13[id-1] = joblib.load(model_path + 'csp_13.pkl')
            self.svm_13[id-1] = joblib.load(model_path + 'svm_13.pkl')
            self.csp_23[id-1] = joblib.load(model_path + 'csp_23.pkl')
            self.svm_23[id-1] = joblib.load(model_path + 'svm_23.pkl')


    def recognize(self, data, personID):
        # print("data:", data)
        # print("data.shape:", data.shape) # 输入数据为59行500列
        # data = data[:, 0:100]
        data -= np.mean(data, axis=1, keepdims=True)
        data_csp_12 = self.fbcsp_transform(data,'12', personID)
        data_csp_13 = self.fbcsp_transform(data,'13', personID)
        data_csp_23 = self.fbcsp_transform(data,'23', personID)
        pro12 = self.svm_12[personID -1].predict_proba(data_csp_12.reshape(1, -1))
        pro13 = self.svm_13[personID -1].predict_proba(data_csp_13.reshape(1, -1))
        pro23 = self.svm_23[personID -1].predict_proba(data_csp_23.reshape(1, -1))
        print("data_csp_12:", data_csp_12.reshape(1,-1))
        print("data_csp_13:", data_csp_13.reshape(1,-1))
        print("data_csp_23:", data_csp_23.reshape(1,-1))
        print("pro12:", pro12)
        print("pro13:", pro13)
        print("pro23:", pro23)
        pro1 = pro12[0, 0] + pro13[0, 0]
        pro2 = pro12[0, 1] + pro23[0, 0]
        pro3 = pro13[0, 1] + pro23[0, 1]
        pro = [pro1, pro2, pro3] 
        result = pro.index(max(pro)) + 201
        return result



