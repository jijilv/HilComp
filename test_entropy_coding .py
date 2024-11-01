import time
import torch
import numpy as np

from utils.encode_utils import entropy_coding
from utils.entropy_model2 import DiscreteUnconditionalEntropyModel, Softmax
from data_compression import rans

# def entropy_coding(attr_index, attr_logits, range_coder_precision=16):
#     # 创建熵模型，并准备 CDF 表
#     em = DiscreteUnconditionalEntropyModel(Softmax(attr_logits), range_coder_precision)
#     em.get_ready_for_compression()

#     # 压缩输入的索引
#     bits_str = em.compress(attr_index)

#     # 解压缩返回的字符串，并检查解压缩结果
#     decode_results = em.decompress(bits_str, attr_index.shape)
    
#     return bits_str, decode_results

def test_entropy_model():
    # 步骤 1：生成随机的 logits 和索引
    logits = torch.randn(65000)  # 生成 65000 个类别的随机 logits
    indexes = torch.randint(0, logits.shape[-1], (3000000,))  # 生成 300 万个随机索引

    # 步骤 2：记录压缩开始时间
    start_time = time.time()

    # 调用 entropy_coding 进行压缩和解压缩
    compressed_string, decompressed_indexes = entropy_coding(indexes, logits)

    # 记录压缩结束时间
    end_time = time.time()

    # 打印压缩时间
    compression_time = end_time - start_time
    print(f"压缩时间: {compression_time:.2f} 秒")

    # 打印解压缩前后的索引是否匹配
    if torch.equal(indexes, decompressed_indexes):
        print("解压缩后的索引匹配！")
    else:
        print("解压缩后的索引不匹配！")

    print("测试完成。")


# 运行测试
test_entropy_model()


# 运行测试
test_entropy_model()

