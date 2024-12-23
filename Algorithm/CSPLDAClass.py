import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import signal
from scipy.signal import butter, filtfilt

rootdir = os.path.dirname(os.path.abspath(__file__))

# 定义多个频率带
frequency_bands = [(8, 12), (12, 30), (30, 50)]

# 仅供测试所用
class CSPLDAClass:
    # def __init__(self,window_size, step_size):
    def __init__(self):
        self.csp_12 = [[None] * len(frequency_bands) for _ in range(6)]
        self.lda_12 = [LDA() for _ in range(6)]
        self.csp_13 = [[None] * len(frequency_bands) for _ in range(6)]
        self.lda_13 = [LDA() for _ in range(6)]
        self.csp_23 = [[None] * len(frequency_bands) for _ in range(6)]
        self.lda_23 = [LDA() for _ in range(6)]
        self.getmodel()
        # self.window_size = 500
        # self.step_size = 500
        # self.window_size = window_size
        # self.step_size = step_size

    def fbcsp_transform(self, data, task, id):
        feature = []
        for band_idx, band in enumerate(frequency_bands):
            if task == '12':
                FBCSP = self.csp_12[id - 1][band_idx]
            elif task == '13':
                FBCSP = self.csp_13[id - 1][band_idx]
            else:
                FBCSP = self.csp_23[id - 1][band_idx]

            Y_prime = FBCSP @ data

            Y_prime -= np.mean(Y_prime, axis=1, keepdims=True)
            band_feature = np.diag(np.cov(Y_prime))
            feature.extend(band_feature)
        return np.array(feature)

    def getmodel(self):
        # 加载训练模型
        for id in range(1, 6):
            model_path = rootdir + '/model/S' + str(id) + '/'
            for band_idx, band in enumerate(frequency_bands):
                self.csp_12[id - 1][band_idx] = joblib.load(model_path + f'csp_12_band{band_idx}.pkl')
                self.lda_12[id - 1] = joblib.load(model_path + f'lda_12.pkl')
                self.csp_13[id - 1][band_idx] = joblib.load(model_path + f'csp_13_band{band_idx}.pkl')
                self.lda_13[id - 1] = joblib.load(model_path + f'lda_13.pkl')
                self.csp_23[id - 1][band_idx] = joblib.load(model_path + f'csp_23_band{band_idx}.pkl')
                self.lda_23[id - 1] = joblib.load(model_path + f'lda_23.pkl')

    def recognize(self, data, personID):  # id为1-5
        
        data -= np.mean(data, axis=1, keepdims=True)
        
        data_csp_12 = self.fbcsp_transform(data, '12', personID).reshape(1, -1)
        data_csp_13 = self.fbcsp_transform(data, '13', personID).reshape(1, -1)
        data_csp_23 = self.fbcsp_transform(data, '23', personID).reshape(1, -1)
        
        # 不再合并所有频率带的特征，而是分别传递给各自的 LDA 模型
        pro12 = self.lda_12[personID - 1].predict_proba(data_csp_12)
        pro13 = self.lda_13[personID - 1].predict_proba(data_csp_13)
        pro23 = self.lda_23[personID - 1].predict_proba(data_csp_23)
        
        # 组合各个 LDA 模型的概率以推断最终结果
        pro1 = pro12[0, 0] + pro13[0, 0]  # P(task1) 的累积概率
        pro2 = pro12[0, 1] + pro23[0, 0]  # P(task2) 的累积概率
        pro3 = pro13[0, 1] + pro23[0, 1]  # P(task3) 的累积概率
        pro = [pro1, pro2, pro3]
        result = pro.index(max(pro)) + 201
        return result


