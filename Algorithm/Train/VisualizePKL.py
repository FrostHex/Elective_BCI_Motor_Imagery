import joblib
import pickle
from numpy.ma.core import shape
import os

# 定义文件路径
# path = '../MI_data_training/S2/block_1.pkl'
# path = './model/S1/lda_12.pkl'
# path = './model/S1/csp_12.pkl'
path = './cov_matrix_task.pkl'

# 加载内容
content = joblib.load(os.path.join(os.path.dirname(__file__), path))
print("content:", content)
print("type:", type(content))
# print(shape(content['ch_names']))

# 遍历所有block的PersonID
# for s in range(1, 6):  # S1 to S5
#     folder = f'../../MI_data_training/S{s}'
#     for b in range(1, 26):  # block_1.pkl to block_25.pkl
#         file = os.path.join(folder, f'block_{b}.pkl')
#         if os.path.exists(file):
#             data = joblib.load(file)
#             print(f"PersonID in {file}: {data['personID']}")

# with open(path, 'rb') as f:
#     loaded_data = pickle.load(f)
# print("Loaded Data:", loaded_data)


###########################################################
# block_1.pkl: 约40秒
# content: {'data': array([[  4949.762207,   4946.007324,   4941.134766, ...,   4978.55127 ,
#           4978.685547,   4976.71875 ],
#        [  7464.06543 ,   7461.695801,   7454.230469, ...,   7599.919434,
#           7597.236816,   7595.314941],
#        [   639.586243,    636.054688,    632.925415, ...,    775.395447,
#            772.355591,    769.226379],
#        ...,
#        [ 40016.117188,  42370.066406,  31285.480469, ...,  23061.828125,
#          32550.857422,  44050.875   ],
#        [-91708.476563, -91712.054688, -91721.125   , ..., -90548.78125 ,
#         -90548.148438, -90547.125   ],
#        [   242.      ,      0.      ,      0.      , ...,      0.      ,
#              0.      ,    243.      ]]),
#        'personID': 1,
#        'blockID': 1,
#        'srate': 250,
#        'ch_names': ['Fpz', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fz', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'FCz', 'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8', 'Cz', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'T7', 'T8', 'CP1', 'CP2', 'CP3', 'CP4', 'CP5', 'CP6', 'TP7', 'TP8', 'Pz', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'POz', 'PO3', 'PO4', 'PO5', 'PO6', 'PO7', 'PO8', 'Oz', 'O1', 'O2', 'ECG', 'HEOR', 'HEOL', 'VEOU', 'VEOL'],
#        'nchan': 64,
#        'dimensions': (65, 9496)}
# type: <class 'dict'>
###########################################################
# lda_12.pkl:
# content: LinearDiscriminantAnalysis()
# type: <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>
###########################################################
# csp_12.pkl:
# content: [CSP({'component_order': 'mutual_info',
#  'cov_est': 'concat',
#  'cov_method_params': None,
#  'log': True,
#  'n_components': 6,
#  'norm_trace': False,
#  'rank': None,
#  'reg': None,
#  'transform_into': 'average_power'}), CSP({'component_order': 'mutual_info',
#  'cov_est': 'concat',
#  'cov_method_params': None,
#  'log': True,
#  'n_components': 6,
#  'norm_trace': False,
#  'rank': None,
#  'reg': None,
#  'transform_into': 'average_power'}), CSP({'component_order': 'mutual_info',
#  'cov_est': 'concat',
#  'cov_method_params': None,
#  'log': True,
#  'n_components': 6,
#  'norm_trace': False,
#  'rank': None,
#  'reg': None,
#  'transform_into': 'average_power'}), CSP({'component_order': 'mutual_info',
#  'cov_est': 'concat',
#  'cov_method_params': None,
#  'log': True,
#  'n_components': 6,
#  'norm_trace': False,
#  'rank': None,
#  'reg': None,
#  'transform_into': 'average_power'}), CSP({'component_order': 'mutual_info',
#  'cov_est': 'concat',
#  'cov_method_params': None,
#  'log': True,
#  'n_components': 6,
#  'norm_trace': False,
#  'rank': None,
#  'reg': None,
#  'transform_into': 'average_power'}), CSP({'component_order': 'mutual_info',
#  'cov_est': 'concat',
#  'cov_method_params': None,
#  'log': True,
#  'n_components': 6,
#  'norm_trace': False,
#  'rank': None,
#  'reg': None,
#  'transform_into': 'average_power'})]
# type: <class 'list'>
###########################################################