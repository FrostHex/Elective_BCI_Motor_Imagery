class CurrentProcessModel:
    def __init__(self):
        # block计数
        self.block_num = 0

        # 当前要读取的数据位置
        self.cur_pos = 0
