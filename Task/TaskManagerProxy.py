from Task.Interface.TaskInterface import TaskInterface


class TaskManagerProxy(TaskInterface):
    """
    该类为赛题TaskManager的一个代理类。
    该类对TaskManager的get_data和report方法进行封装
    将该类传给AlgorithmInterface，可以避免算法端调用赛题的其他方法
    """
    def __init__(self, task_mng):
        self.__task_mng = task_mng

    def get_data(self):
        return self.__task_mng.get_data()

    def report(self, report_model):
        self.__task_mng.report(report_model)
