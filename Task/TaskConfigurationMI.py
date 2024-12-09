class TaskConfigurationMI:

    def __init__(self):
        # 目标采样率(Hz)，题目说明中给出
        self.samp_rate = 250

        # 单次数据包时长(s)，题目说明中给出
        # pkg_len * samp_rate必须为正整数
        self.pkg_time = 0.04

        # 单试次最长判决时间，题目中说明给出，超过该值，则认为判断错误。
        self.max_trial_time = 5

        # 备选目标数，题目说明中给出
        self.target_num = 3

        # 事件id
        event_id = []
        # 添加刺激事件id
        for stim_event_id in range(1, 4):
            event_id.append(stim_event_id)
        # 添加trial开始事件id
        event_id.append(4)

        # 事件
        event = []
        # 添加刺激事件
        event= [201, 202, 203]
        # 添加trial开始事件
        event.append(240)

        # 事件类型
        event_type = []
        # 添加刺激事件类型
        for _ in [201, 202, 203]:
            event_type.append('STIMULATION')
        # 添加trial开始事件类型
        event_type.append('TRIALSTART')
        # 刺激事件频率

        # 事件定义列表，题目说明中给出
        self.event_tbl = {
            'id': event_id,
            'event': event,
            'type': event_type,
        }


if __name__ == '__main__':
    task_config = TaskConfigurationMI()
    print(task_config.event_tbl)
