import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from mne.decoding import CSP
from scipy import signal

# Trigger 定义
# 实验开始	实验结束	 Block开始	Block结束	Trial开始	Trial结束	左手想象	右手想象	双脚想象	测试集特有(想象开始)
#  250        251     242         243         240         241         201     202     203     249

class Train():
    def __init__(self):
        self.rootdir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../MI_data_training/'))
        # Load your training data
        # X_train: EEG data of shape (trials, channels, samples)
        # y_train: Labels of shape (trials,)
        return

    def run(self):
        data = self.get_data(1,1)
        data = self.test_overview(data,250)
        trigger = self.get_data(1,1,65) # 第65通道为trigger
        trigger = self.test_overview(trigger,250)
        # print("trigger:", trigger)
        # label = self.get_label(1,2)
        # print("label:", label)
        # print(len(label))
        return

    # @brief: 读取pkl文件内数据
    # @param: id: 从1到5,代表S1到S5，5位受试者
    # @param: block: 从1到25，代表第1到第25个block数据
    # @param: channel: 从1到65，代表第1到第65个通道数据
    # @return: 返回数据
    def get_data(self, id, block, channel=None):
        data = joblib.load(self.rootdir + '/S' + str(id) + '/block_' + str(block) + '.pkl')['data']
        if channel is not None:
            return data[channel - 1]
        else:
            return data

    def get_label(self, id, block):
        temp = self.get_data(id, block, 65)
        label = [x for x in temp if x in [201, 202, 203]]
        return label

    def test_overview(self, data, downsampling=None):
        print("original shape:",data.shape)
        if data.ndim == 1: data = data[::downsampling]
        if data.ndim == 2: data = data[:,::downsampling]
        print(data)
        if (downsampling is not None): print("down sampled shape:", data.shape)
        return data


if __name__ == '__main__':
    train = Train()
    train.run()