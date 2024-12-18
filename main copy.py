import sys
import os

sys.path.append('.')

from Algorithm.AlgorithmImplementMI import AlgorithmImplementMI
from Framework.AlgorithmSystemManager import AlgorithmSystemManager
from Task.TaskManagerProxy import TaskManagerProxy
from Task.TaskManagerMI import TaskManagerMI
from Framework.loadData import load_data


if __name__ == '__main__':
    best_score = 0
    best_accuracy = 0
    best_accuracy_window_size = 0
    best_accuracy_step_size = 0
    best_score_window_size = 0
    best_score_step_size = 0
    for window_size in range(60, 80, 1):
        for step_size in range(130, 210, 1):
            # 系统框架实例
            algorithm_sys_mng = AlgorithmSystemManager()
            # MI赛题实例
            task_mng = TaskManagerMI()
            # 向系统框架注入MI赛题
            algorithm_sys_mng.add_task(task_mng)
            # 创建该赛题的代理对象
            task_mng_proxy = TaskManagerProxy(task_mng)
            # 向系统框架注入赛题代理对象
            algorithm_sys_mng.add_task_proxy(task_mng_proxy)
            # 加载MI数据
            data_path = os.path.join(os.getcwd(), 'TestData')
            mi_data_path = os.path.join(data_path, 'MI')
            dataPath = os.environ.get('DATASET', mi_data_path)
            # 读取所有被试者的MI数据
            subject_data_model_set = load_data(dataPath)
            # 向赛题注入MI数据
            algorithm_sys_mng.add_data(subject_data_model_set)
            # 算法实例
            algorithm_impl_mi = AlgorithmImplementMI(window_size, step_size)
            # # 向系统框架注入算法
            algorithm_sys_mng.add_algorithm(algorithm_impl_mi)
            # # 执行算法
            algorithm_sys_mng.run()
            # # 获取评分
            scoreModel,accuracy = algorithm_sys_mng.get_score()
            # # 处理结果
            if scoreModel.score > best_score:
                best_score = scoreModel.score
                best_score_window_size = window_size
                best_score_step_size = step_size
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_accuracy_window_size = window_size
                best_accuracy_step_size = step_size
            print("trying: ", window_size,",", step_size, ". score: ", scoreModel.score, "accuracy: ", accuracy)
            print("best score: ", best_score, "with: ", best_score_window_size, ",", best_score_step_size)
            print("best accuracy: ", best_accuracy, "with: ", best_accuracy_window_size, ",", best_accuracy_step_size)



            # 清除算法
            algorithm_sys_mng.clear_algorithm()
            # 清除数据
            algorithm_sys_mng.clear_data()
            # 清除赛题
            algorithm_sys_mng.clear_task()
