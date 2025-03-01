# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule, kaiming_init
from mmcv.runner import load_checkpoint

from ...utils import cache_checkpoint, get_root_logger
from ..builder import BACKBONES
import torch

@BACKBONES.register_module()
class C3D(nn.Module):
    """C3D backbone, without flatten and mlp.

    Args:
        pretrained (str | None): Name of pretrained model.
    """

    def __init__(self,
                 in_channels=3,
                base_channels=64,
                num_stages=4,
                temporal_downsample=True,
                pretrained=None):
        super().__init__()

        # 配置卷积、归一化和激活函数参数
        conv_cfg = dict(type='Conv3d')
        norm_cfg = dict(type='BN3d')
        act_cfg = dict(type='ReLU')

        # 初始化预训练模型
        self.pretrained = pretrained

        # 输入通道数
        self.in_channels = in_channels

        # 基础通道数
        self.base_channels = base_channels

        # 断言：num_stages 必须是 3 或 4
        assert num_stages in [3, 4]

        # 初始化 num_stages
        self.num_stages = num_stages

        # 是否进行时间维度下采样
        self.temporal_downsample = temporal_downsample

        # 初始化池化核大小和步长
        pool_kernel, pool_stride = 2, 2

        # 如果不进行时间维度下采样，则修改池化核大小和步长
        if not self.temporal_downsample:
            pool_kernel, pool_stride = (1, 2, 2), (1, 2, 2)

        # 初始化 C3D 卷积参数
        c3d_conv_param = dict(kernel_size=3, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # 第一层卷积和池化
        self.conv1a = ConvModule(self.in_channels, self.base_channels, **c3d_conv_param)
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # 第二层卷积和池化
        self.conv2a = ConvModule(self.base_channels, self.base_channels * 2, **c3d_conv_param)
        self.pool2 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        # 第三层卷积和池化
        self.conv3a = ConvModule(self.base_channels * 2, self.base_channels * 4, **c3d_conv_param)
        self.conv3b = ConvModule(self.base_channels * 4, self.base_channels * 4, **c3d_conv_param)
        self.pool3 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)

        # 第四层卷积
        self.conv4a = ConvModule(self.base_channels * 4, self.base_channels * 8, **c3d_conv_param)
        self.conv4b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

        # 如果 num_stages 为 4，则添加第五层卷积和池化
        if self.num_stages == 4:
            self.pool4 = nn.AvgPool3d(kernel_size=pool_kernel, stride=pool_stride)
            self.conv5a = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)
            self.conv5b = ConvModule(self.base_channels * 8, self.base_channels * 8, **c3d_conv_param)

    def init_weights(self):
        """
        从现有的检查点或从头开始初始化参数。
        """
        # 遍历模型中的所有模块
        for m in self.modules():
            # 如果模块是3D卷积层
            if isinstance(m, nn.Conv3d):
                # 使用kaiming初始化方法对卷积层进行初始化
                kaiming_init(m)

        # 如果pretrained属性是字符串类型
        if isinstance(self.pretrained, str):
            # 获取根日志记录器
            logger = get_root_logger()
            # 记录日志，输出加载模型的路径
            logger.info(f'load model from: {self.pretrained}')
            # 从缓存中获取检查点
            self.pretrained = cache_checkpoint(self.pretrained)
            # 加载检查点，并将参数加载到当前模型中
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

    def forward(self, x):
        """定义了每次调用时执行的计算。

        Args:
            x (torch.Tensor): 输入数据。x 的大小是 (num_batches, 3, 16, 112, 112)。

        Returns:
            torch.Tensor: 通过骨干网络提取的输入样本的特征。
        """
        # 第一层卷积
        x = self.conv1a(x)

        # 第一层池化
        x = self.pool1(x)
        # 第二层卷积
        x = self.conv2a(x)

        # 第二层池化
        x = self.pool2(x)
        # 第三层卷积1
        x = self.conv3a(x)
        # 第三层卷积2
        x = self.conv3b(x)

        # 第三层池化
        x = self.pool3(x)
        print(x.shape)
        # 第四层卷积1
        x = self.conv4a(x)
        # 第四层卷积2
        x = self.conv4b(x)

        # 如果网络阶段数为3，则返回当前特征
        if self.num_stages == 3:
            return x

        # 第四层池化
        x = self.pool4(x)
        # 第五层卷积1
        x = self.conv5a(x)
        # 第五层卷积2
        x = self.conv5b(x)

        return x

# 实例化C3D模型
c3d_model = C3D(in_channels=3, base_channels=64, num_stages=4, temporal_downsample=True)

# 假设我们有一个批次大小为1的输入张量，其形状为(num_batches, 3, 16, 112, 112)
# 这里的3代表RGB通道，16代表视频帧数，112, 112代表空间维度（高度和宽度）
input_tensor = torch.randn(1, 3, 16, 112, 112)

# 将模型设置为评估模式
c3d_model.eval()

# 通过模型传递输入张量以获取输出
with torch.no_grad():
    output_tensor = c3d_model(input_tensor)

# 打印输出张量的形状
print(output_tensor.shape)