from abc import ABCMeta, abstractmethod


class TaskInterface(metaclass=ABCMeta):
    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def report(self, report_model):
        pass
