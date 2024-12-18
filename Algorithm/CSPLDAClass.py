import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import signal

rootdir = os.path.dirname(os.path.abspath(__file__))


# 仅供测试所用
class CSPLDAClass:
    def __init__(self):
        self.csp_12 = [None] * 6
        self.lda_12 = [LDA() for _ in range(6)]
        self.csp_13 = [None] * 6
        self.lda_13 = [LDA() for _ in range(6)]
        self.csp_23 = [None] * 6
        self.lda_23 = [LDA() for _ in range(6)]
        self.getmodel()

    def fbcsp_transform(self, data, task, id):
        if task == '12':
            FBCSP = self.csp_12[id - 1]
        elif task == '13':
            FBCSP = self.csp_13[id - 1]
        else:
            FBCSP = self.csp_23[id - 1]

        data -= np.mean(data, axis=1, keepdims=True)
        Y_prime = FBCSP @ data
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
        # print("data:", data)
        # print("data.shape:", data.shape) # 输入数据为59行500列
        # data = data[:, 0:100]

        data_csp_12 = self.fbcsp_transform(data, '12', personID)
        data_csp_13 = self.fbcsp_transform(data, '13', personID)
        data_csp_23 = self.fbcsp_transform(data, '23', personID)
        pro12 = self.lda_12[personID - 1].predict_proba(data_csp_12.reshape(1, -1))
        pro13 = self.lda_13[personID - 1].predict_proba(data_csp_13.reshape(1, -1))
        pro23 = self.lda_23[personID - 1].predict_proba(data_csp_23.reshape(1, -1))
        print("data_csp_12:", data_csp_12.reshape(1, -1))
        print("data_csp_13:", data_csp_13.reshape(1, -1))
        print("data_csp_23:", data_csp_23.reshape(1, -1))
        print("pro12:", pro12)
        print("pro13:", pro13)
        print("pro23:", pro23)
        pro1 = pro12[0, 0] + pro13[0, 0]
        pro2 = pro12[0, 1] + pro23[0, 0]
        pro3 = pro13[0, 1] + pro23[0, 1]
        pro = [pro1, pro2, pro3]
        result = pro.index(max(pro)) + 201
        return result


