import abc, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from data_compression._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from data_compression import rans
except:
    pass

def pmf_to_quantized_cdf(pmf, precision: int = 16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class Softmax:
    def __init__(self, logits):
        super().__init__()
        self.patch_shape = logits.shape[:-1]
        self.pmf_length = logits.shape[-1]
        self.logits = logits

    def prob(self, indexes):
        pmf = F.softmax(self.logits, dim=-1)
        prob = pmf.gather(indexes, dim=-1)
        if indexes.requires_grad:
            return torch.clamp_min(prob, math.log(1e-9))
        else:
            return prob

    def log_prob(self, indexes):
        log_pmf = F.log_softmax(self.logits, dim=-1)
        log_prob = log_pmf.gather(indexes, dim=-1)
        if indexes.requires_grad:
            return torch.clamp_min(log_prob, math.log(1e-9))
        else:
            return log_prob

    def pmf(self):
        pmf = F.softmax(self.logits, dim=-1)

        if pmf.requires_grad:
            return torch.clamp_min(pmf, math.log(1e-9))
        else:
            return pmf

    def log_pmf(self):
        log_pmf = F.log_softmax(self.logits, dim=-1)

        if log_pmf.requires_grad:
            return torch.clamp_min(log_pmf, math.log(1e-9))
        else:
            return log_pmf

class DiscreteEntropyModelBase(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self,
                 prior=None,
                 tail_mass=2**-8,
                 range_coder_precision=16):
        super().__init__()
        self._prior = prior
        self._tail_mass = float(tail_mass)
        self._range_coder_precision = int(range_coder_precision)
        try:
            self._encoder = rans.RansEncoder()
            self._decoder = rans.RansDecoder()
        except:
            pass
        self.register_buffer("_cdf_shape", torch.IntTensor(2).zero_())

    @property
    def prior(self):
        """Prior distribution, used for deriving range coding tables."""
        if self._prior is None:
            raise RuntimeError(
            "This entropy model doesn't hold a reference to its prior "
            "distribution.")
        return self._prior

    @property
    def tail_mass(self):
        return self._tail_mass

    @property
    def range_coder_precision(self):
        return self._range_coder_precision

    def _log_prob_from_prior(self, prior, indexes):
        return prior.log_prob(indexes)

    def _log_pmf_from_prior(self, prior):
        return prior.log_pmf()

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)

    def compress(self, indexes, cdf_indexes):
        # 将输入张量转换为 Python 列表
        symbols_list = indexes.flatten().int().tolist()
        indexes_list = cdf_indexes.flatten().int().tolist()

        # 确保 self.cdf 是一个二维列表
        if isinstance(self.cdf, torch.Tensor):
            cdf_list = self.cdf.tolist()  # 如果是张量，直接转换为列表
        elif isinstance(self.cdf, list):
            if isinstance(self.cdf[0], list):
                cdf_list = self.cdf  # 已经是列表的列表
            else:
                cdf_list = [self.cdf]  # 如果是单层列表，转换为二维列表
        else:
            raise ValueError("Invalid CDF format: must be a tensor or list of lists")

        # 将 CDF 长度和偏移量转换为一维列表
        if isinstance(self.cdf_length, torch.Tensor):
            cdf_length_list = self.cdf_length.flatten().tolist()
        else:
            cdf_length_list = list(self.cdf_length)  # 转换为一维列表

        if isinstance(self.cdf_offset, torch.Tensor):
            cdf_offset_list = self.cdf_offset.flatten().tolist()
        else:
            cdf_offset_list = list(self.cdf_offset)  # 转换为一维列表

        # 调用 encode_with_indexes()，确保所有参数类型正确
        string = self._encoder.encode_with_indexes(
            symbols_list,
            indexes_list,
            cdf_list,
            cdf_length_list,
            cdf_offset_list,
        )

        return string

    def decompress(self, string, cdf_indexes):
        # 获取设备和形状信息
        device = cdf_indexes.device
        shape = cdf_indexes.shape

        # 将 cdf_indexes 转换为一维整数列表
        cdf_indexes = cdf_indexes.flatten().int().tolist()

        # 确保 self.cdf 是一个二维列表
        if isinstance(self.cdf, torch.Tensor):
            cdf_list = self.cdf.tolist()  # 如果是张量，转换为二维列表
        elif isinstance(self.cdf, list):
            if isinstance(self.cdf[0], list):
                cdf_list = self.cdf  # 如果已经是列表的列表，直接使用
            else:
                cdf_list = [self.cdf]  # 如果是单层列表，转换为二维列表
        else:
            raise ValueError("Invalid CDF format: must be a tensor or list of lists")

        # 将 CDF 长度和偏移量转换为一维列表
        if isinstance(self.cdf_length, torch.Tensor):
            cdf_length_list = self.cdf_length.flatten().tolist()
        else:
            cdf_length_list = list(self.cdf_length)  # 转换为一维列表

        if isinstance(self.cdf_offset, torch.Tensor):
            cdf_offset_list = self.cdf_offset.flatten().tolist()
        else:
            cdf_offset_list = list(self.cdf_offset)  # 转换为一维列表

        # 调用 decode_with_indexes()，确保所有参数类型正确
        try:
            values = self._decoder.decode_with_indexes(
                string,
                cdf_indexes,
                cdf_list,
                cdf_length_list,
                cdf_offset_list,
            )

            # 将解压后的值转换为原始张量形状
            values = torch.tensor(values, device=device, dtype=torch.int64).reshape(shape)
            return values

        except Exception as e:
            print("解压失败:", str(e))
            raise

    def get_ready_for_compression(self):
        self._init_tables()
        self._fix_tables()
        self.cdf = self._cdf.int().tolist()
        self.cdf_length = self._cdf_length.flatten().int().tolist()
        self.cdf_offset = self._cdf_offset.flatten().int().tolist()

    def _fix_tables(self):
        cdf, cdf_offset, cdf_length = self._build_tables(
            self.prior, self.range_coder_precision)

        self._cdf_shape.data = torch.IntTensor(list(cdf.shape)).to(cdf.device)
        self._init_tables()
        self._cdf.data = cdf.int()
        self._cdf_offset.data = torch.IntTensor([cdf_offset])
        self._cdf_length.data = torch.IntTensor([cdf_length])

    def _init_tables(self):
        shape = self._cdf_shape.tolist()
        device = self._cdf_shape.device
        self.register_buffer("_cdf", torch.IntTensor(*shape).to(device))
        self.register_buffer("_cdf_offset", torch.IntTensor(*shape[:1]).to(device))
        self.register_buffer("_cdf_length", torch.IntTensor(*shape[:1]).to(device))

    def _build_tables(self, prior, precision):
        pmf = prior.pmf()
        pmf_length = pmf.shape[0]

        cdf_length = pmf_length + 2
        cdf_offset = 0

        max_length = pmf_length

        cdf = torch.zeros([max_length + 2], dtype=torch.int32)

        # Adjust for a single PMF without batch dimension
        overflow = (1. - pmf.sum(dim=0, keepdim=True)).clamp_min(0)
        pmf = torch.cat([pmf, overflow], dim=0)
        c = pmf_to_quantized_cdf(pmf, precision)

        # Fill the CDF table for a single PMF
        cdf[:c.shape[0]] = c

        return cdf, cdf_offset, cdf_length

class DiscreteUnconditionalEntropyModel(DiscreteEntropyModelBase):

    def __init__(self,
                 prior,
                 range_coder_precision=16):
        super().__init__(
            prior=prior,
            range_coder_precision=range_coder_precision)

    def forward(self, indexes):
        log_probs = self._log_prob_from_prior(self.prior, indexes)
        bits = torch.sum(log_probs) / (-math.log(2))
        return bits

    def log_pmf(self,):
        return self._log_pmf_from_prior(self.prior)

    @staticmethod
    def _build_cdf_indexes(shape):
        # 获取 CDF 的长度 C
        C = shape[0]

        # 生成一个全为 0 的索引张量
        indexes = torch.zeros(C, dtype=torch.int)

        return indexes

    def compress(self, indexes):
        cdf_indexes = self._build_cdf_indexes(indexes.shape)
        return super().compress(indexes, cdf_indexes)

    def decompress(self, string, shape):
        cdf_indexes = self._build_cdf_indexes(shape)
        return super().decompress(string, cdf_indexes)