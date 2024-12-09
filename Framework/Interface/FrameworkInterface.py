from abc import ABCMeta, abstractmethod


class FrameworkInterface(metaclass=ABCMeta):

    # 填充任务
    @abstractmethod
    def add_task(self, task_manager):
        pass

    # 填充数据
    @abstractmethod
    def add_data(self, person_data_transfer_model_set):
        pass

    # 填充算法
    @abstractmethod
    def add_algorithm(self, algorithm_interface):
        pass

    # 运行算法
    @abstractmethod
    def run(self):
        pass

    # 取得成绩
    @abstractmethod
    def get_score(self):
        pass

    # 清空已有数据
    @abstractmethod
    def clear_data(self):
        pass

    # 清除当前算法所有结果，为下一个算法做准备
    @abstractmethod
    def clear_task(self):
        pass

    # 清除算法
    @abstractmethod
    def clear_algorithm(self):
        pass
