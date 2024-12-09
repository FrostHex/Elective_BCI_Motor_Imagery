from Framework.Interface.FrameworkInterface import FrameworkInterface
from Task.TaskConfigurationMI import TaskConfigurationMI


class AlgorithmSystemManager(FrameworkInterface):

    def __init__(self):
        # 赛题
        self.task_mng = None

        # 赛题代理
        self.task_mng_proxy = None

        # 算法
        self.algorithm_impl = None

    # 填充赛题
    def add_task(self, task_mng):
        self.task_mng = task_mng
        # 若为SSVEP赛题
        if self.task_mng.PARADIGMNAME == 'SSVEP':
            pass
        # 若为ERP赛题
        elif self.task_mng.PARADIGMNAME == 'ERP':
            pass
        # 若为MI赛题
        elif self.task_mng.PARADIGMNAME == 'MI':
            task_config_mi = TaskConfigurationMI()
            self.task_mng.initial(task_config_mi)
        # 若为EMOTION赛题
        elif self.task_mng.PARADIGMNAME == 'EMOTION':
            pass

    # 填充数据
    def add_data(self, subject_data_model_set):
        self.task_mng.add_data(subject_data_model_set)

    # 填充算法
    def add_algorithm(self, algorithm_impl):
        self.algorithm_impl = algorithm_impl
        self.algorithm_impl.set_task(self.task_mng_proxy)
        self.task_mng.init_record()

    # 运行算法
    def run(self):
        self.algorithm_impl.run()

    # 取得成绩
    def get_score(self):
        score_model = self.task_mng.get_score()
        return score_model

    # 清除赛题
    def clear_task(self):
        self.task_mng = None

    # 清空已有数据
    def clear_data(self):
        self.task_mng.clear_data()

    # 清除当前算法所有结果，为下一个算法做准备
    def clear_algorithm(self):
        self.algorithm_impl = None
        self.task_mng.clear_record()
    
    def add_task_proxy(self, task_mng_proxy):
        self.task_mng_proxy = task_mng_proxy
