# import torch
# from BiLstm import * 
# class Config:
#     def __init__(self):

#         # 模型配置
#         self.lstm_hidden_size = 256
#         self.dense_hidden_size = 2048
#         self.embed_size = 2048
#         self.num_layers = 2
# dict_path = f'C:\\Users\\wuxin\\Desktop\\KD_test\\KD_test\\CorrNet\\preprocess\\phoenix2014\\gloss_dict.npy'  # Use the gloss dict of phoenix14 dataset 
# gloss_dict = np.load(dict_path, allow_pickle=True).item()
# gloss_dict= dict((v[0], k) for k, v in gloss_dict.items())   

# torch_model = BiLSTM_SA_temp(Config(),len(gloss_dict)+1) 					# 由研究员提供python.py文件
# checkpoint = torch.load('C:\\Users\\wuxin\\Desktop\\KD_test\\kd_test.pth')
# torch_model.load_state_dict(checkpoint['model_state_dict'])
# batch_size = 1 								# 批处理大小
# input_shape = (200, 276) 				# 输入数据
 
# # set the model to inference mode
# torch_model.eval()
 
# x = torch.randn(batch_size,*input_shape) 	# 生成张量
# export_onnx_file = "test.onnx" 				# 目的ONNX文件名
# torch.onnx.export(torch_model,
#                     x,
#                     export_onnx_file,
#                     opset_version=10,
#                     do_constant_folding=True,	# 是否执行常量折叠优化
#                     input_names=["input"],		# 输入名
#                     output_names=["output"],	# 输出名
#                     dynamic_axes={"input":{0:"batch_size"},	# 批处理变量
#                                     "output":{0:"batch_size"}})
import sys
sys.path.append("C:\\Users\\wuxin\\Desktop\\KD_test\\test.onnx")  # onnx2tflite的地址

from converter import onnx_converter
onnx_path = "C:\\Users\\wuxin\\Desktop\\KD_test\\test.onnx"  # 需要转换的onnx文件位置
onnx_converter(
    onnx_model_path = onnx_path,
    need_simplify = True,
    output_path = "C:\\Users\\wuxin\\Desktop\\KD_test\\",  # 输出的tflite存储路径
    target_formats = ['tflite'], # or ['keras'], ['keras', 'tflite']
    weight_quant = False,
    int8_model = False,
    int8_mean = None,
    int8_std = None,
    image_root = None
)
