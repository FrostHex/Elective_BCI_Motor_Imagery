import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy import signal
import warnings
warnings.filterwarnings("ignore")


rootdir = os.path.dirname(os.path.abspath(__file__))
# 仅供测试所用
class CSPLDAClass:
    def fbcsp_transform(self, X_train, FBCSP):
        x = np.zeros([X_train.shape[0], 0]) # 初始化空数组，用于存储变换后的数据
        filteeg = np.zeros(np.shape(X_train)) # 初始化滤波后的EEG数据数组
        bandVue = np.array([[1, 8], [8, 16], [16, 24], [24, 32], [32, 40], [40, 48]]) # 定义频带范围
        for i in range(6): # 对每个频带进行处理
            # 设计带通滤波器
            b, a = signal.butter(4, [2 * bandVue[i, 0] / 250, 2 * bandVue[i, 1] / 250], 'bandpass', analog=True)
            # 对每个试次进行滤波
            for trail in range(X_train.shape[0]):
                for ch in range(X_train.shape[1]): # 对每个通道进行E滤波
                    filteeg[trail, ch, :] = signal.filtfilt(b, a, X_train[trail, ch, :])
            # 使用CSP变换滤波后的数据
            print("filteeg:", filteeg)
            print("filteeg.shape:", filteeg.shape)
            print("type:", type(filteeg))
            XFB_train = FBCSP[i].transform(filteeg)
            # 将变换后的数据拼接到结果数组中
            x = np.concatenate((x, XFB_train), axis=1)
        return x

    def getmodel(self, personID):
        # 加载训练模型
        model_path = rootdir + '/model/S' + str(personID) + '/'

        csp_12 = joblib.load(model_path + 'csp_12.pkl')
        lda_12 = joblib.load(model_path + 'lda_12.pkl')

        csp_13 = joblib.load(model_path + 'csp_13.pkl')
        lda_13 = joblib.load(model_path + 'lda_13.pkl')

        csp_23 = joblib.load(model_path + 'csp_23.pkl')
        lda_23 = joblib.load(model_path + 'lda_23.pkl')
        return [csp_12, lda_12, csp_13, lda_13, csp_23, lda_23]

    def recognize(self, data, personID):
        #data = self.band_Filter(data, personID)  # 每个人选择不同的带通滤波器
        mod = self.getmodel(personID)
        csp_12 = mod[0]
        lda_12 = mod[1]
        csp_13 = mod[2]
        lda_13 = mod[3]
        csp_23 = mod[4]
        lda_23 = mod[5]

        # print("data:", data)
        # print("data.shape:", data.shape)

        data_csp_12 = self.fbcsp_transform(np.expand_dims(data, 0), csp_12)
        data_csp_13 = self.fbcsp_transform(np.expand_dims(data, 0), csp_13)
        data_csp_23 = self.fbcsp_transform(np.expand_dims(data, 0), csp_23)
        # data_csp_12 = csp_12.transform(np.expand_dims(data, 0))
        # data_csp_13 = csp_13.transform(np.expand_dims(data, 0))
        # data_csp_23 = csp_23.transform(np.expand_dims(data, 0))
        pro12 = lda_12.predict_proba(data_csp_12)
        pro13 = lda_13.predict_proba(data_csp_13)
        pro23 = lda_23.predict_proba(data_csp_23)
        pro1 = pro12[0, 0] + pro13[0, 0]
        pro2 = pro12[0, 1] + pro23[0, 0]
        pro3 = pro13[0, 1] + pro23[0, 1]
        pro = [pro1, pro2, pro3] #
        result = pro.index(max(pro)) + 201
        return result



