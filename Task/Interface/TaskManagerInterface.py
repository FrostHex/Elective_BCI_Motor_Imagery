from abc import ABCMeta, abstractmethod
from Task.Interface.TaskInterface import TaskInterface


class TaskManagerInterface(TaskInterface):

    @abstractmethod
    def initial(self, task_configuration):
        pass

    @abstractmethod
    def add_data(self, person_data_transfer_model_set):
        pass

    @abstractmethod
    def get_score(self):
        pass

    @abstractmethod
    def clear_data(self):
        pass

    @abstractmethod
    def clear_record(self):
        pass

    @abstractmethod
    def init_record(self):
        pass
