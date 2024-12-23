import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import signal
from scipy.signal import butter, filtfilt

rootdir = os.path.dirname(os.path.abspath(__file__))


# 仅供测试所用
class CSPLDAClass:
    # def __init__(self,window_size, step_size):
    def __init__(self):
        self.csp_12 = [None] * 6
        self.lda_12 = [LDA() for _ in range(6)]
        self.csp_13 = [None] * 6
        self.lda_13 = [LDA() for _ in range(6)]
        self.csp_23 = [None] * 6
        self.lda_23 = [LDA() for _ in range(6)]
        self.getmodel()
        # self.window_size = 500
        # self.step_size = 500
        # self.window_size = window_size
        # self.step_size = step_size

    def fbcsp_transform(self, data, task, id):
        if task == '12':
            FBCSP = self.csp_12[id - 1]
        elif task == '13':
            FBCSP = self.csp_13[id - 1]
        else:
            FBCSP = self.csp_23[id - 1]

        Y_prime = FBCSP @ data

        b1, a1 = butter(4, [8, 15], btype='bandpass', fs=250)
        b2, a2 = butter(4, [18, 24], btype='bandpass', fs=250)
        Y_prime = filtfilt(b1, a1, Y_prime, axis=1) + filtfilt(b2, a2, Y_prime, axis=1)
        Y_prime -= np.mean(Y_prime, axis=1, keepdims=True)
        feature = np.diag(np.cov(Y_prime))
        return feature

    def getmodel(self):
        # 加载训练模型
        for id in range(1, 6):
            model_path = rootdir + '/model/S' + str(id) + '/'
            self.csp_12[id - 1] = joblib.load(model_path + 'csp_12.pkl')
            self.lda_12[id - 1] = joblib.load(model_path + 'lda_12.pkl')
            self.csp_13[id - 1] = joblib.load(model_path + 'csp_13.pkl')
            self.lda_13[id - 1] = joblib.load(model_path + 'lda_13.pkl')
            self.csp_23[id - 1] = joblib.load(model_path + 'csp_23.pkl')
            self.lda_23[id - 1] = joblib.load(model_path + 'lda_23.pkl')

    def recognize(self, data, personID):  # id为1-5
        
        data -= np.mean(data, axis=1, keepdims=True)
        
        data_csp_12 = self.fbcsp_transform(data, '12', personID)
        data_csp_13 = self.fbcsp_transform(data, '13', personID)
        data_csp_23 = self.fbcsp_transform(data, '23', personID)
        
        pro12 = self.lda_12[personID - 1].predict_proba(data_csp_12.reshape(1, -1))
        pro13 = self.lda_13[personID - 1].predict_proba(data_csp_13.reshape(1, -1))
        pro23 = self.lda_23[personID - 1].predict_proba(data_csp_23.reshape(1, -1))
        
        pro1 = pro12[0, 0] + pro13[0, 0]
        pro2 = pro12[0, 1] + pro23[0, 0]
        pro3 = pro13[0, 1] + pro23[0, 1]
        pro = [pro1, pro2, pro3]
        result = pro.index(max(pro)) + 201
        return result


