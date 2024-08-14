import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

TF_PATH = "C:\\Users\\wuxin\\Desktop\\KD_test\\test.pb"  # 转tensorflow的pb文件的存储路径
ONNX_MODEL_PATH = "C:\\Users\\wuxin\\Desktop\\KD_test\\test.onnx"  # 需要转换的onnx文件路径
ONNX_MODEL_simple_PATH = "C:\\Users\\wuxin\\Desktop\\KD_test\\"  # 需要转换的onnx文件路径
WORKING_DIR = "C:\\Users\\wuxin\\Desktop\\KD_test\\"
TFLITE_PATH = "C:\\Users\\wuxin\\Desktop\\KD_test\\test.tflite"  # 输出的tflite文件路径

import onnxsim
import onnx

simplified_onnx_model, success = onnxsim.simplify(ONNX_MODEL_PATH)
assert success, 'Failed to simplify the ONNX model. You may have to skip this step'
simplified_onnx_model_path =  f'{ONNX_MODEL_simple_PATH}simplified.onnx'

print(f'Generating {simplified_onnx_model_path} ...')
onnx.save(simplified_onnx_model, simplified_onnx_model_path)
print('done')

import onnx
from onnx_tf.backend import prepare
from onnx2keras import onnx_to_keras
import keras
import tensorflow as tf


def onnx_to_pb(output_path):
    '''
    将.onnx模型保存为.pb文件模型
    '''
    model = onnx.load(output_path) #加载.onnx模型文件
    tf_rep = prepare(model)
    tf_rep.export_graph('model_all.pb')    #保存最终的.pb文件

def onnx_to_h5(output_path ):
    '''
    将.onnx模型保存为.h5文件模型,并打印出模型的大致结构
    '''
    onnx_model = onnx.load(output_path)
    k_model = onnx_to_keras(onnx_model, ['input'])
    keras.models.save_model(k_model, 'kerasModel.h5', overwrite=True, include_optimizer=True)    #第二个参数是新的.h5模型的保存地址及文件名
    # 下面内容是加载该模型，然后将该模型的结构打印出来
    model = tf.keras.models.load_model('kerasModel.h5')
    model.summary()
    print(model)
    
if __name__=='__main__':
    input_path = "C:\\Users\\wuxin\\Desktop\\KD_test\\test.onnx"    #输入需要转换的.pth模型路径及文件名
    output_path = "model_all.onnx"  #转换为.onnx后文件的保存位置及文件名
   
    # onnx_pre(output_path)   #【可选项】若有需要，可以使用onnxruntime进行部署测试，看所转换模型是否可用，其中，output_path指加载进去的onnx格式模型所在路径及文件名
    # onnx_to_pb(output_path)   #将onnx模型转换为pb模型
    onnx_to_h5(output_path )   #将onnx模型转换为h5模型
