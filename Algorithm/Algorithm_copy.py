from Algorithm.Interface.AlgorithmInterface import AlgorithmInterface
from Algorithm.Interface.Model.ReportModel import ReportModel
from Algorithm.CSPLDAClass import CSPLDAClass
from scipy import signal
import numpy as np
import math


class AlgorithmImplementMI(AlgorithmInterface):
    # 类属性：范式名称
    PARADIGMNAME = 'MI'

    def __init__(self):
        super().__init__()
        # 定义采样率，题目文件中给出
        self.samp_rate = 250
        # 选择导联编号
        self.select_channel = list(range(1, 60))
        self.select_channel = [i - 1 for i in self.select_channel]
        # 想象开始的trigger（由于240作为trial开始被占用，这里使用系统预留trigger:249）
        self.trial_stimulate_mask_trig = 249
        # trial结束trigger
        self.trial_end_trig = 241
        # 定义最小和最大计算时间
        self.min_cal_time = 2  # 最小2秒
        self.max_cal_time = 2  # 最大2秒
        # 最小和最大计算长度
        self.min_cal_len = self.min_cal_time * self.samp_rate
        self.max_cal_len = self.max_cal_time * self.samp_rate
        # 初始化方法
        self.method = CSPLDAClass()
        # 预处理滤波器设置
        self.filterB, self.filterA = self.__get_pre_filter(self.samp_rate)
        # 初始化缓存
        self.__clear_cache()

    def run(self):
        # 是否停止标签
        end_flag = False
        # 是否进入计算模式标签
        cal_flag = False
        while not end_flag:
            data_model = self.task.get_data()
            if not cal_flag:
                # 非计算模式，则进行事件检测
                cal_flag = self.__idle_proc(data_model)  # 检测trigger
            else:
                # 计算模式，则进行处理
                cal_flag, result = self.__cal_proc(data_model)  # 处理入口
                # 如果有结果，则进行报告
                if result is not None:
                    report_model = ReportModel()
                    report_model.result = result
                    self.task.report(report_model)
                    # 清空缓存
                    self.__clear_cache()
            end_flag = data_model.finish_flag

    def __idle_proc(self, data_model):
        # 脑电数据+trigger
        data = data_model.data
        # 获取trigger导
        trigger = data[-1, :]
        trigger_idx = np.where(trigger == self.trial_stimulate_mask_trig)[0]
        # 脑电数据
        eeg_data = data[0: -1, :]
        if len(trigger_idx) > 0:
            # 有trial开始trigger则进入计算模式
            cal_flag = True
            trial_start_trig_pos = trigger_idx[0]
            # 从trial开始的位置拼接数据
            self.cache_data = eeg_data[:, trial_start_trig_pos:]
        else:
            # 没有trial开始trigger
            cal_flag = False
            self.__clear_cache()
        return cal_flag

    def __cal_proc(self, data_model):
        # 脑电数据+trigger
        data = data_model.data
        personID = data_model.subject_id
        # 获取trigger导
        trigger = data[-1, :]
        # 获取脑电数据
        eeg_data = data[0: -1, :]
        # 将新数据添加到缓存
        self.cache_data = np.append(self.cache_data, eeg_data, axis=1)
        # 检查是否有新的trial开始
        trigger_idx = np.where(trigger == self.trial_stimulate_mask_trig)[0]
        # 计算动态窗口长度
        cal_len = self.__calculate_dynamic_window(self.cache_data)
        current_len = self.cache_data.shape[1]

        # 如果缓存数据长度达到或超过动态窗口长度，进行计算
        if current_len >= cal_len:
            # 获取需要的数据
            use_data = self.cache_data[:, :int(cal_len)]
            # 滤波处理
            use_data = self.__preprocess(use_data)
            # 开始计算
            result = self.method.recognize(use_data, personID)
            # 停止计算模式
            cal_flag = False
        else:
            result = None
            cal_flag = True

        # 如果检测到新的trial开始trigger，重置缓存
        if len(trigger_idx) > 0:
            # 清除缓存的数据
            self.__clear_cache()
            # 开始新trial的数据缓存
            next_trial_start_trig_pos = trigger_idx[0]
            self.cache_data = eeg_data[:, next_trial_start_trig_pos:]
            cal_flag = True

        return cal_flag, result

    def __calculate_dynamic_window(self, data):
        # 动态计算窗口长度，可以根据数据特征进行调整
        # 这里以数据的方差为例进行动态调整
        variance = np.var(data)
        # 将方差映射到[min_cal_len, max_cal_len]之间
        cal_len = self.min_cal_len + (self.max_cal_len - self.min_cal_len) * (variance / (variance + 1))
        # 限制cal_len在[min_cal_len, max_cal_len]范围内
        cal_len = np.clip(cal_len, self.min_cal_len, self.max_cal_len)
        return cal_len

    def __get_pre_filter(self, samp_rate):
        fs = samp_rate
        f0 = 50
        q = 35
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        return b, a

    def __clear_cache(self):
        self.cache_data = np.zeros((len(self.select_channel), 0))

    def __preprocess(self, data):
        # 选择使用的导联
        data = data[self.select_channel, :]
        filter_data = signal.filtfilt(self.filterB, self.filterA, data)  # scipy滤波函数 零相移滤波
        return filter_data