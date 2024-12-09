import os
import pickle
import random
import numpy as np
from Framework.Interface.Model.BlockDataModel import BlockDataModel
from Framework.Interface.Model.SubjectDataModel import SubjectDataModel


def load_data(folder_path):
    # 所有被试对应的目录
    subject_dirs = []
    # 遍历文件夹，获取各个被试对应的目录名
    for root, dirs, files in os.walk(folder_path):
        if dirs:
            subject_dirs = dirs
            subject_dirs.sort()
            break
    # 所有被试的数据模型集合
    subject_data_model_set = []
    # 对于每个被试
    for subject_dir in subject_dirs:
        # 该被试对应数据的路径
        subject_data_path = os.path.join(folder_path, subject_dir)
        # 该被试的所有block的数据模型的集合
        block_data_model_set = []
        # 遍历文件夹，获取该被试目录下的所有文件
        for root, dirs, files in os.walk(subject_data_path):
            if files:
                files.sort()
                print('被试{}的目录下的所有文件：'.format(subject_dir) + str(files))
                # 遍历所有文件，找到所有.pkl结尾的文件
                for file in files:
                    if str(file).endswith('.pkl'):
                        # 打开.pkl文件
                        pkl_file = open(os.path.join(subject_data_path, file), "rb")
                        # 加载该block的eeg数据模型
                        eeg = pickle.load(pkl_file)
                        # 获取该block的数据
                        data = eeg['data']
                        # down_samp_data = down_sample(data)
                        # 获取该block的id
                        block_name = file[0: str(file).find('.')]
                        # 该block的数据模型
                        block_data_model = BlockDataModel(block_name, data)
                        # 将该block的数据模型添加到被试的block数据集合中
                        block_data_model_set.append(block_data_model)
        # 该被试的数据模型
        subject_data_model = SubjectDataModel(subject_dir, block_data_model_set)
        # 所有被试的数据模型集合
        subject_data_model_set.append(subject_data_model)
    return subject_data_model_set   #(name, block_data_model_set)


# def down_sample(data):
#     trigger = data[-1, :]
#     trigger_num = len(trigger[trigger != 0])
#     if trigger_num == 0:
#         down_sample_data = data[:, ::4]  # 降采样至1/4
#         return down_sample_data
#     # print("该数据包中有" + str(trigger_num) + "个trigger")
#     down_sample_data = data[:-1, ::4]  # 降采样至1/4
#     trigger_index = np.where(trigger != 0)[0]
#     triggerData = np.zeros((1, down_sample_data.shape[1]))
#     for i in trigger_index:
#         trig = trigger[i]
#         down_sample_trigger_index = int(i / 4)
#         triggerData[0][down_sample_trigger_index] = trig
#     down_sample_data = np.concatenate((down_sample_data, triggerData), axis=0)
#     return down_sample_data


if __name__ == '__main__':
    path = os.path.join(os.path.dirname(os.getcwd()), '2022_bci_competition_preliminary-main\TestData', 'MI')
    print(path)
    subject_data_model_set = load_data(path)
    print(subject_data_model_set[0].block_data_model_set[0].data.shape)

