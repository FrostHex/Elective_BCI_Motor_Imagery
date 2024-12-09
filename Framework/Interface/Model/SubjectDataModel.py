class SubjectDataModel:
    def __init__(self, subject_name, block_data_model_set):
        # 被试名
        self.name = subject_name

        # 该被试所有block数据模型的集合
        self.block_data_model_set = block_data_model_set
