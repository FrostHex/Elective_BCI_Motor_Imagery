import os

import joblib
import numpy as np
from mne.decoding import CSP
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit

from Framework.loadData import load_data


def fbcsp_fit_transform(traindata, y_train, FBCSP):
    x = np.zeros([traindata.shape[0], 0])
    filteeg = np.zeros(np.shape(traindata))
    bandVue = np.array([[1, 8], [8, 16], [16, 24], [24, 32], [32, 40], [40, 48]])
    for i in range(6):
        b, a = signal.butter(4, [2 * bandVue[i, 0] / 250, 2 * bandVue[i, 1] / 250], 'bandpass', analog=True)
        for trail in range(traindata.shape[0]):
            for ch in range(traindata.shape[1]):
                filteeg[trail, ch, :] = signal.filtfilt(b, a, traindata[trail, ch, :])
        XFB_train = FBCSP[i].fit_transform(filteeg, y_train)  # csp空间滤波器训练
        x = np.concatenate((x, XFB_train), axis=1)
    return x, FBCSP


def fbcsp_transform(X_train, FBCSP):
    x = np.zeros([X_train.shape[0], 0])
    filteeg = np.zeros(np.shape(X_train))
    bandVue = np.array([[1, 8], [8, 16], [16, 24], [24, 32], [32, 40], [40, 48]])
    for i in range(6):
        b, a = signal.butter(4, [2 * bandVue[i, 0] / 250, 2 * bandVue[i, 1] / 250], 'bandpass', analog=True)
        for trail in range(X_train.shape[0]):
            for ch in range(X_train.shape[1]):
                filteeg[trail, ch, :] = signal.filtfilt(b, a, X_train[trail, ch, :])
        XFB_train = FBCSP[i].transform(filteeg)  # csp空间滤波器训练
        x = np.concatenate((x, XFB_train), axis=1)
    return x


def mi_trian(data_list1, data_list2):
    epochdata1 = data_list1[0]
    epochdata2 = data_list2[0]
    label1 = data_list1[1]
    label2 = data_list2[1]
    epochdata = np.concatenate((epochdata1, epochdata2), axis=0)
    labels = np.concatenate((label1, label2), axis=0)

    # csp共同空间模式
    lda = LinearDiscriminantAnalysis()
    FBCSP = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        FBCSP[i] = CSP(n_components=6, reg=None, log=True, norm_trace=False)

    # 通过交叉验证来获得最佳的分类器
    scores = []
    traindataset = []
    traindatasetid = []

    cv = ShuffleSplit(10, test_size=0.2)
    # cv = ShuffleSplit(10, test_size=0.2, random_state=42)
    cv_split = cv.split(epochdata)
    for train_idx, test_idx in cv_split:
        y_train, y_test = labels[train_idx], labels[test_idx]

        traindata = epochdata[train_idx]
        X_train, FBCSP = fbcsp_fit_transform(traindata, y_train, FBCSP)  # csp空间滤波器训练

        lda.fit(X_train, y_train)  # 线性分类器训练
        X_test = fbcsp_transform(epochdata[test_idx], FBCSP)
        # X_test = csp.transform(epochdata[test_idx])  # 测试集特征提取
        scores.append(lda.score(X_test, y_test))
        traindataset.append(traindata)
        traindatasetid.append(y_train)

    # 获得最佳的性能的分类器参数
    if len(scores) > 0:  # 获得了分类器
        mid = np.argsort(scores)[-1]
        X_train, FBCSP = fbcsp_fit_transform(traindataset[mid], traindatasetid[mid], FBCSP)
        # X_train = csp.fit_transform(traindataset[mid], traindatasetid[mid])
        lda.fit(X_train, traindatasetid[mid])

        print("============================================")
        print("最佳分类器性能")
        print("使用了{}组训练数据".format(epochdata.shape[0]))
        print("分类正确率为：{}".format(scores[mid]))
        print("============================================")
        print("scores: {}".format(scores))
        print("average scores {}".format(sum(scores) / len(scores)))

        return FBCSP, lda

    else:
        raise Exception('更新分类器失败')


if __name__ == '__main__':
    def __preprocess(data):
        # 选择使用的导联
        fs = 250
        f0 = 50
        q = 35
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        data = data[0:59, :]
        filter_data = signal.filtfilt(b, a, data)  # scipy滤波函数 零相移滤波
        return filter_data


    data_path = os.path.join(os.getcwd(), 'TestData', 'MI')
    cal_time = 2
    sample_rate = 250
    sample_point_nb = cal_time * sample_rate
    data_trigger = np.zeros(90)

    # 载入数据
    subject_data_model_set = load_data(data_path)
    for k in range(5):
        trigger = np.zeros(0)
        data = np.zeros([0, 59, int(sample_point_nb)])
        S01 = subject_data_model_set[k]

        for j in range(2):
            block = S01.block_data_model_set[j + 1]
            trigger_zero = block.data[-1, :]
            for i in range(len(trigger_zero)):
                if block.data[-1, i] == 201 or block.data[-1, i] == 202 or block.data[-1, i] == 203:
                    # print(data[-1, i])
                    trigger = np.append(trigger, block.data[-1, i])
                    data_cache = block.data[0:64, i:i + int(sample_point_nb)]
                    data_cache = __preprocess(data_cache)
                    data_c = data_cache[None, :, :]
                    data = np.append(data, data_c, 0)
        label1 = np.repeat(201, 30)
        label2 = np.repeat(202, 30)
        label3 = np.repeat(203, 30)
        data1 = data[trigger == 201, :, :]
        data2 = data[trigger == 202, :, :]
        data3 = data[trigger == 203, :, :]
        datalist1 = [data1, label1]
        datalist2 = [data2, label2]
        datalist3 = [data3, label3]

        mitrain_path = os.path.abspath('.')

        # 训练模型
        csp_12, lda_12 = mi_trian(datalist1, datalist2)
        csp_13, lda_13 = mi_trian(datalist1, datalist3)
        csp_23, lda_23 = mi_trian(datalist2, datalist3)
        # 保存模型
        cspmodel_path = os.path.join(mitrain_path, 'Algorithm', 'model', 'S' + str(k + 1))
        ldamodel_path = os.path.join(mitrain_path, 'Algorithm', 'model', 'S' + str(k + 1))
        joblib.dump(csp_12, cspmodel_path + '\csp_12.pkl')
        joblib.dump(lda_12, ldamodel_path + '\lda_12.pkl')
        joblib.dump(csp_13, cspmodel_path + '\csp_13.pkl')
        joblib.dump(lda_13, ldamodel_path + '\lda_13.pkl')
        joblib.dump(csp_23, cspmodel_path + '\csp_23.pkl')
        joblib.dump(lda_23, ldamodel_path + '\lda_23.pkl')
