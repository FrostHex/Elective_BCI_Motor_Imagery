from Task.Interface.TaskManagerInterface import TaskManagerInterface
from Task.Model.CurrentProcessModel import CurrentProcessModel
from Task.Interface.Model.ScoreModel import ScoreModel
from Algorithm.Interface.Model.DataModel import DataModel
from Task.Model.TrialFlagModel import TrialFlagModel
import numpy as np
import math
from random import shuffle, sample


class TaskManagerMI(TaskManagerInterface):
    # 范式名称
    PARADIGMNAME = "MI"

    def __init__(self):
        # 受试者信息
        self.subject_tbl = {"id": [], "name": []}  # 该受试者id  # 该受试者名
        # 以block为单位存储数据
        self.block_data_tbl = {
            "id": [],  # 该block数据对应的id
            "subject_id": [],  # 该block数据对应的受试者
            "block_name": [],  # 该block的名
            "data": [],  # 该block的数据
        }
        # 打乱的block_data_tbl
        self.shuffle_block_data_tbl = {"id": [], "subject_id": [], "block_name": [], "data": []}
        # 试次信息列表
        self.trial_tbl = {
            "id": [],  # 该trial对应的id
            "block_id": [],  # 该trial所对应的block的id
            "trigger": [],  # 该trial所对应的trigger
            "trigger_position": [],  # trigger在该block的数据中的位置
            "subject_id": [],  # 该trial所对应的被试id
        }
        self.shuffle_trial_tbl = {"id": [], "block_id": [], "trigger": [], "trigger_position": [], "subject_id": []}
        # 结果记录信息
        self.record_tbl = {
            "id": [],  # 该次汇报结果的id
            "block_id": [],  # 该次汇报结果对应block的id
            "report_point": [],  # 该次汇报时已读取到的数据的位置
            "result": [],  # 该次汇报的结果
        }
        # 每个数据包的长度
        self.pkg_len = 0
        # 试次数据最大读取长度
        self.max_trial_len = 0
        # 采样率
        self.samp_rate = 0
        # 目标数
        self.target_num = 0
        # 刺激trigger编号列表
        self.stim_trig = []
        # trial开始trigger编号
        self.trial_start_trig = None
        # 赛题配置
        self.task_config = None
        # 当前进度模型
        self.cur_proc_model = CurrentProcessModel()  # data_file_table_id = 1，cur_pos = 0
        # trial是否开始的检验位
        self.trial_flag_model = TrialFlagModel()

    def initial(self, task_config):
        # 初始化赛题配置
        self.task_config = task_config
        event_tbl = self.task_config.event_tbl
        # 定义各种事件类型的trigger
        for event_idx in range(len(event_tbl["id"])):
            # 刺激事件trigger
            if event_tbl["type"][event_idx] == "STIMULATION":
                self.stim_trig.append(event_tbl["event"][event_idx])
            # trial开始trigger
            elif event_tbl["type"][event_idx] == "TRIALSTART":
                self.trial_start_trig = event_tbl["event"][event_idx]
        # 采样率
        self.samp_rate = self.task_config.samp_rate
        # 刺激目标数量
        self.target_num = self.task_config.target_num
        # 数据包的长度 = 数据包时长 * 采样率
        self.pkg_len = self.task_config.pkg_time * self.task_config.samp_rate
        # 试次数据最大读取长度 = 试次最大时长 * 采样率
        self.max_trial_len = self.task_config.max_trial_time * self.task_config.samp_rate

    # 添加所有被试的所有block的数据
    def add_data(self, subject_data_model_set):
        # subject_data_model_set包含两个元素：name（被试名），block_data_model_set（被试数据）
        # 遍历所有被试
        id_list = [id+1 for id in range(len(subject_data_model_set))]
        choice_subject_id = sample(id_list, 1)
        for subject_data_model in subject_data_model_set:
            # 获取被试名
            subject_name = subject_data_model.name
            # 获取该被试所有block的数据模型
            block_data_model_set = subject_data_model.block_data_model_set
            # 添加被试并获取其id
            subject_id = self.__add_subject(subject_name)
            if subject_id in choice_subject_id and len(block_data_model_set)==2:
                shuffle_block_data_model = block_data_model_set[0]
                block_data_model_set[0] = block_data_model_set[1]
                block_data_model_set[1] = shuffle_block_data_model
            # 遍历该被试所有block的数据
            for block_data_model in block_data_model_set:
                self.__add_block_data(subject_id, block_data_model.name, block_data_model.data)

    # 获取得分
    def get_score(self):
        # 调用函数
        score_model, accuracy = self.__cal_score()
        return score_model, accuracy

    # 清除数据
    def clear_data(self):
        self.subject_tbl = {"id": [], "name": []}
        self.block_data_tbl = {"id": [], "subject_id": [], "block_name": [], "data": []}
        self.trial_tbl = {"id": [], "block_id": [], "trigger": [], "trigger_position": [], "subject_id": []}
        self.clear_record()

    def clear_record(self):
        self.record_tbl = {"id": [], "block_id": [], "report_point": [], "result": []}
        self.cur_proc_model.data_file_table_id = 1
        self.cur_proc_model.currentPosition = 0

    # 打乱被试
    def init_record(self):
        # 打乱被试者id
        subject_id = [sid for sid in self.subject_tbl["id"]]
        shuffle(subject_id)
        # 打乱trial_tbl
        for sid in subject_id:
            for idx, id_ in enumerate(self.trial_tbl["subject_id"]):
                if sid == id_:
                    self.shuffle_trial_tbl["id"].append(self.trial_tbl["id"][idx])
                    self.shuffle_trial_tbl["block_id"].append(self.trial_tbl["block_id"][idx])
                    self.shuffle_trial_tbl["trigger_position"].append(self.trial_tbl["trigger_position"][idx])
                    self.shuffle_trial_tbl["trigger"].append(self.trial_tbl["trigger"][idx])
                    self.shuffle_trial_tbl["subject_id"].append(self.trial_tbl["subject_id"][idx])
        # 打乱block_data_tbl
        for sid in subject_id:
            for idx, id_ in enumerate(self.block_data_tbl["subject_id"]):
                if sid == id_:
                    self.shuffle_block_data_tbl["id"].append(self.block_data_tbl["id"][idx])
                    self.shuffle_block_data_tbl["subject_id"].append(self.block_data_tbl["subject_id"][idx])
                    self.shuffle_block_data_tbl["block_name"].append(self.block_data_tbl["block_name"][idx])
                    self.shuffle_block_data_tbl["data"].append(self.block_data_tbl["data"][idx])

    # 算法调用该方法获取数据
    def get_data(self):
        # 生成一个data模型
        data_model = DataModel()
        shuffle_block_data_tbl = self.shuffle_block_data_tbl
        # block编号
        block_id = shuffle_block_data_tbl["id"][self.cur_proc_model.block_num]
        # 受试者编号
        subject_id = shuffle_block_data_tbl["subject_id"][self.cur_proc_model.block_num]
        # 数据读取的开始位置，从0开始
        start_pt = self.cur_proc_model.cur_pos
        # 数据读取的结束位置
        end_pt = start_pt + self.pkg_len
        # 定义脑电数据
        eeg_data = None
        # 取出对应的block的数据
        for idx, bid in enumerate(shuffle_block_data_tbl["id"]):
            if block_id == bid:
                eeg_data = shuffle_block_data_tbl["data"][idx]
        # 该block脑电数据的长度
        eeg_data_len = eeg_data.shape[1]
        # block是否结束的标志位
        block_end_flag = False
        # 数据读取的起始点最小为0
        if start_pt < 0:
            start_pt = 0
        # 数据读取的结束点最大为该block的数据长度
        if end_pt > eeg_data_len:
            end_pt = eeg_data_len
            # 该block的数据已读完
            block_end_flag = True
        # 读取起始点到结束点的数据
        data = eeg_data[:, int(start_pt) : int(end_pt)]
        for data_idx in range(data.shape[1]):
            if data[-1, data_idx] in self.stim_trig:
                data[-1, data_idx] = 249
                self.trial_flag_model.trial_flag = False
                data[:-1, :data_idx] = 0
                break
            if data[-1, data_idx] == 240:
                self.trial_flag_model.trial_flag = True
                data[:-1, data_idx:] = 0
                break
            if self.trial_flag_model.trial_flag:
                data[:-1, data_idx] = 0
        self.cur_proc_model.cur_pos = self.cur_proc_model.cur_pos + data.shape[1]
        # 是否停止的标志位
        finish_flag = False
        # 如果该block结束
        if block_end_flag:
            # block计数加1
            self.cur_proc_model.block_num = self.cur_proc_model.block_num + 1
            # 是否为最后一个block
            if self.cur_proc_model.block_num >= len(shuffle_block_data_tbl["id"]):
                finish_flag = True
            else:
                # 更新当前要读取的数据的位置
                self.cur_proc_model.cur_pos = 0
        # 填充data模型
        data_model.data = data
        data_model.start_pos = start_pt
        data_model.subject_id = subject_id
        data_model.finish_flag = finish_flag
        return data_model

    # 算法调用该方法报告结果
    def report(self, report_model):
        # 算法反馈结果
        result = report_model.result
        # 当前block的id
        block_id = self.shuffle_block_data_tbl["id"][self.cur_proc_model.block_num]
        # 当前block已读取到的数据的位置
        cur_pos = self.cur_proc_model.cur_pos
        # 本次报告结果的id
        record_id = len(self.record_tbl["id"]) + 1
        self.record_tbl["id"].append(record_id)
        self.record_tbl["block_id"].append(block_id)
        self.record_tbl["report_point"].append(cur_pos)
        self.record_tbl["result"].append(result)

    # 添加受试者
    def __add_subject(self, name):
        subject_id = len(self.subject_tbl["id"]) + 1
        self.subject_tbl["id"].append(subject_id)
        self.subject_tbl["name"].append(name)
        return subject_id

    # 以block为单位添加数据，并保存该block中所有trial的信息
    def __add_block_data(self, subject_id, block_name, block_data):
        # 添加该被试某个block的数据
        block_data_id = len(self.block_data_tbl["id"]) + 1
        self.block_data_tbl["id"].append(block_data_id)
        self.block_data_tbl["subject_id"].append(subject_id)
        self.block_data_tbl["block_name"].append(block_name)
        self.block_data_tbl["data"].append(block_data)
        # 保存该block中所有trial的信息
        # 该block数据的trigger通道
        trig_channel = block_data[-1, :]
        # 所有trigger的位置
        trig_pos = np.nonzero(trig_channel)[0]
        # 获取当前trial id
        trial_id = len(self.trial_tbl["id"]) + 1
        # 遍历所有的trigger
        for pos in trig_pos:
            # 通过trigger所在位置的索引获取trigger的值
            trig = trig_channel[pos]
            # 判断是否为刺激类型trigger
            if trig in self.stim_trig:
                self.trial_tbl["id"].append(trial_id)
                self.trial_tbl["block_id"].append(block_data_id)
                self.trial_tbl["trigger"].append(int(trig))
                self.trial_tbl["trigger_position"].append(pos)
                self.trial_tbl["subject_id"].append(subject_id)
                trial_id = trial_id + 1

    # 计算得分
    def __cal_score(self):
        shuffle_trial_tbl = self.shuffle_trial_tbl
        # 所有被试的所有block的所有trial的trigger位置
        trig_pos = []
        for pos in shuffle_trial_tbl["trigger_position"]:
            trig_pos.append(pos)
        # 该trigger所在数据包(pkg_len长度)的最后一个数据点的位置
        # 记录该位置是因为算法需要读取至少一个数据包再给出汇报结果，所以算法针对该trial给出结果时使用的数据长度即'report_point'的值应大于该位置的值
        trig_pkg_end_pos = []
        for pos in trig_pos:
            trig_pkg_end_pos.append(self.pkg_len * math.ceil(pos / self.pkg_len))
        # 汇报结果集合
        result = []
        # 汇报时长集合（存储的是单个trial计算出结果所用的数据长度）
        report_time_len = []

        # 对于每个trial，最后一个trial单独计算
        for trial_idx in range(len(shuffle_trial_tbl["id"]) - 1):
            # 当前trial的block的id
            cur_block_id = shuffle_trial_tbl["block_id"][trial_idx]
            # 下一个trial的block的id
            next_block_id = shuffle_trial_tbl["block_id"][trial_idx + 1]
            # 临时存储汇报结果
            temp_record_tbl = {"id": [], "block_id": [], "report_point": [], "result": []}
            # 遍历汇报结果，找出该block的所有汇报结果
            for idx, bid in enumerate(self.record_tbl["block_id"]):
                if bid == cur_block_id:
                    temp_record_tbl["id"].append(self.record_tbl["id"][idx])
                    temp_record_tbl["block_id"].append(self.record_tbl["block_id"][idx])
                    temp_record_tbl["report_point"].append(self.record_tbl["report_point"][idx])
                    temp_record_tbl["result"].append(self.record_tbl["result"][idx])
            # 该trial报告结果所在位置处的索引
            record_idx = -1
            # 如果下一个试次不是同一个block文件，寻找报告点大于当前trigger位置的第一次报告
            if cur_block_id != next_block_id:
                for idx, report_point in enumerate(temp_record_tbl["report_point"]):
                    if report_point > trig_pkg_end_pos[trial_idx]:
                        record_idx = idx
                        break
                if record_idx >= 0:
                    # 当前trial给出汇报结果
                    cur_trail_result = temp_record_tbl["result"][record_idx]
                    cur_trial_report_time_len = temp_record_tbl["report_point"][record_idx] - trig_pos[trial_idx]
                else:
                    # 当前trial未给出汇报结果，设置反馈结果为0
                    cur_trail_result = 0
                    cur_trial_report_time_len = self.max_trial_len + 1
            else:
                # 如果下一个试次位于同一个block中
                # 寻找报告点大于当前trigger位置，且小于等于下一次trigger位置的第一个汇报点
                for idx, report_point in enumerate(temp_record_tbl["report_point"]):
                    if trig_pkg_end_pos[trial_idx] < report_point < trig_pkg_end_pos[trial_idx + 1]:
                        record_idx = idx
                        break
                if record_idx >= 0:
                    # 当前trial给出汇报结果
                    cur_trail_result = temp_record_tbl["result"][record_idx]
                    cur_trial_report_time_len = temp_record_tbl["report_point"][record_idx] - trig_pos[trial_idx]
                else:
                    # 当前trial未给出汇报结果，设置反馈结果为0
                    cur_trail_result = 0
                    cur_trial_report_time_len = self.max_trial_len + 1
            # 将当前trial结果添加到结果列表中
            result.append(cur_trail_result)
            # 将当前trial所用时间长度添加到汇报时长列表中
            report_time_len.append(cur_trial_report_time_len)

        # 最后一个trial单独处理
        block_id = shuffle_trial_tbl["block_id"][-1]
        last_trial_record_tbl = {"id": [], "block_id": [], "report_point": [], "result": []}
        for idx, bid in enumerate(self.record_tbl["block_id"]):
            if bid == block_id:
                last_trial_record_tbl["id"].append(self.record_tbl["id"][idx])
                last_trial_record_tbl["block_id"].append(self.record_tbl["block_id"][idx])
                last_trial_record_tbl["report_point"].append(self.record_tbl["report_point"][idx])
                last_trial_record_tbl["result"].append(self.record_tbl["result"][idx])
        # 该trial报告结果所在位置处的索引
        record_idx = -1
        # 找到第一个大于最后一个trigger位置的报告点
        for idx, report_point in enumerate(last_trial_record_tbl["report_point"]):
            if report_point > trig_pkg_end_pos[-1]:
                record_idx = idx
                break
        if record_idx >= 0:
            cur_trail_result = last_trial_record_tbl["result"][record_idx]
            cur_trial_report_time_len = last_trial_record_tbl["report_point"][record_idx] - trig_pos[-1]
        else:
            cur_trail_result = 0
            cur_trial_report_time_len = self.max_trial_len + 1
        result.append(cur_trail_result)
        report_time_len.append(cur_trial_report_time_len)

        # 报告时间（使用计算所用的数据长度折算为计算时间）
        report_time = []
        for idx, report_len in enumerate(report_time_len):
            report_time.append(report_len / self.samp_rate)
            if report_len > self.max_trial_len:
                result[idx] = 0
        accuracy_for_one_block_set = []
        # 统计正确数
        # 以block为单位计算ITR及正确率
        block_id = [bid for bid in shuffle_trial_tbl["block_id"]]
        # 每个block id 只保留一次
        block_id_set = set(block_id)
        block_id = [bid for bid in block_id_set]
        block_id.sort()
        block_itr = []
        # 对于每个block
        for bid in block_id:
            # 报告结果
            block_result = [result[idx] for idx, id_ in enumerate(shuffle_trial_tbl["block_id"]) if id_ == bid]
            # 原本结果
            block_trigger = [
                shuffle_trial_tbl["trigger"][idx] for idx, id_ in enumerate(shuffle_trial_tbl["block_id"]) if id_ == bid
            ]
            # 报告时间
            cur_block_trial_report_time = [
                report_time[idx] for idx, id_ in enumerate(shuffle_trial_tbl["block_id"]) if id_ == bid
            ]
            # 统计判断正确的数量
            correct_num = 0
            for i in range(len(block_trigger)):
                if block_result[i] == block_trigger[i]:
                    correct_num += 1
            # 平均预测时间
            cur_block_avg_trial_report_time = sum(cur_block_trial_report_time) / len(cur_block_trial_report_time)
            # 预测准确率
            accuracy_for_one_block = correct_num / len(block_result)
            accuracy_for_one_block_set.append(accuracy_for_one_block)
            # 计算该block的ITR
            itr = 60 * self.__cal_itr(self.target_num, accuracy_for_one_block, cur_block_avg_trial_report_time)
            block_itr.append(itr)
        score_model = ScoreModel()
        score_model.score = sum(block_itr) / len(block_itr)
        print("accuracy:", np.average(accuracy_for_one_block_set))
        return score_model, np.average(accuracy_for_one_block_set)

    # 计算ITR
    def __cal_itr(self, N, P, T, margin=0.5):
        if P <= 1 / N:
            ITR = 0
        elif P == 1:
            ITR = (1 / (T+margin)) * (math.log(N, 2) + P * math.log(P, 2))
        else:
            ITR = (1 / (T+margin)) * (math.log(N, 2) + (1 - P) * math.log((1 - P) / (N - 1), 2) + P * math.log(P, 2))
        return ITR

    # def __calculateITR(self, N, P, T):

    #     if P == 0:
    #         ITR = (1 / T) * (math.log(N, 2) + (1 - P) * math.log((1 - P) / (N - 1), 2))
    #     if P <= 1 / N:
    #         ITR = 0
    #     elif P == 1:
    #         ITR = (1 / T) * (math.log(N, 2) + P * math.log(P, 2))
    #     else:
    #         ITR = (1 / T) * (math.log(N, 2) + (1 - P) * math.log((1 - P) / (N - 1), 2) + P * math.log(P, 2))
    #     return ITR
