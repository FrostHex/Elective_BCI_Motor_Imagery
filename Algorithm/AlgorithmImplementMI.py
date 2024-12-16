from Algorithm.Interface.AlgorithmInterface import AlgorithmInterface
from Algorithm.Interface.Model.ReportModel import ReportModel
from Algorithm.CSPLDAClass import CSPLDAClass
from scipy import signal
import numpy as np

class AlgorithmImplementMI(AlgorithmInterface):
    PARADIGMNAME = 'MI'

    def __init__(self):
        super().__init__()
        samp_rate = 250
        self.samp_rate = samp_rate
        self.select_channel = [i - 1 for i in range(1, 60)]
        self.trial_stimulate_mask_trig = 249
        self.trial_end_trig = 241
        cal_time = 2
        self.cal_len = cal_time * samp_rate
        self.method = CSPLDAClass()
        self.filterB, self.filterA = self.__get_pre_filter(samp_rate)
        self.cache_data = np.zeros((64, self.cal_len))
        self.cache_pos = 0

    def run(self):
        end_flag = False
        cal_flag = False
        while not end_flag:
            data_model = self.task.get_data()
            if not cal_flag:
                cal_flag = self.__idle_proc(data_model)
            else:
                cal_flag, result = self.__cal_proc(data_model)
                if result is not None:
                    report_model = ReportModel()
                    report_model.result = result
                    self.task.report(report_model)
                    self.__clear_cache()
            end_flag = data_model.finish_flag

    def __idle_proc(self, data_model):
        data = data_model.data
        trigger = data[-1, :]
        trigger_idx = np.where(trigger == self.trial_stimulate_mask_trig)[0]
        eeg_data = data[0:-1, :]
        if len(trigger_idx) > 0:
            cal_flag = True
            trial_start_trig_pos = trigger_idx[0]
            self.cache_pos = 0
            data_slice = eeg_data[:, trial_start_trig_pos:]
            copy_len = min(data_slice.shape[1], self.cal_len)
            self.cache_data[:, :copy_len] = data_slice[:, :copy_len]
            self.cache_pos += copy_len
        else:
            cal_flag = False
            self.__clear_cache()
        return cal_flag

    def __cal_proc(self, data_model):
        data = data_model.data
        personID = data_model.subject_id
        trigger = data[-1, :]
        trigger_idx = np.where(trigger == self.trial_stimulate_mask_trig)[0]
        eeg_data = data[0:-1, :]
        if len(trigger_idx) == 0:
            remaining_len = self.cal_len - self.cache_pos
            copy_len = min(eeg_data.shape[1], remaining_len)
            self.cache_data[:, self.cache_pos:self.cache_pos + copy_len] = eeg_data[:, :copy_len]
            self.cache_pos += copy_len
            if self.cache_pos >= self.cal_len:
                use_data = self.__preprocess(self.cache_data)
                result = self.method.recognize(use_data, personID)
                cal_flag = False
                self.cache_pos = 0
            else:
                result = None
                cal_flag = True
        else:
            next_trial_start_trig_pos = trigger_idx[0]
            remaining_len = self.cal_len - self.cache_pos
            copy_len = min(next_trial_start_trig_pos, remaining_len)
            self.cache_data[:, self.cache_pos:self.cache_pos + copy_len] = eeg_data[:, :copy_len]
            self.cache_pos += copy_len
            use_data = self.__preprocess(self.cache_data[:, :self.cache_pos])
            result = self.method.recognize(use_data, personID)
            cal_flag = True
            self.cache_pos = 0
            data_slice = eeg_data[:, next_trial_start_trig_pos:]
            copy_len = min(data_slice.shape[1], self.cal_len)
            self.cache_data[:, :copy_len] = data_slice[:, :copy_len]
            self.cache_pos += copy_len
        return cal_flag, result

    def __get_pre_filter(self, samp_rate):
        fs = samp_rate
        f0 = 50
        q = 35
        b, a = signal.iircomb(f0, q, ftype='notch', fs=fs)
        return b, a

    def __clear_cache(self):
        self.cache_pos = 0

    def __preprocess(self, data):
        data = data[self.select_channel, :]
        filter_data = signal.filtfilt(self.filterB, self.filterA, data)
        return filter_data
