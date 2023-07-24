import mindspore as ms
import numpy as np
import time
from mindspore import nn, Parameter
from mindspore import Tensor
from mindspore.train import Model
from mindspore.dataset import GeneratorDataset
from mindspore.profiler import Profiler
from utils_mae.monitor import LossMonitorMAE
from utils_mae.logger import get_logger


ms.set_context(mode=ms.GRAPH_MODE) # "PYNATIVE_MODE" "GRAPH_MODE"
ms.set_context(device_target="Ascend")
ms.set_context(device_id=0)
logger = get_logger('./output/')

DATA_TYPE = 'image' #'video','image'

if DATA_TYPE == 'video':
    #train_dataset1:模拟视频数据传入windowattention的特征(64, 392, 128)
    train_dataset = GeneratorDataset(source=[(np.random.randn(64, 392, 128),),
                                             (np.random.randn(64, 392, 128),),
                                             (np.random.randn(64, 392, 128),),
                                             (np.random.randn(64, 392, 128),)],
                                     column_names=["col"])
elif DATA_TYPE == 'image':
    #train_dataset2:模拟图像数据传入windowattention的特征(64, 49, 128)
    train_dataset = GeneratorDataset(source=[(np.random.randn(64, 49, 128),),
                                             (np.random.randn(64, 49, 128),),
                                             (np.random.randn(64, 49, 128),),
                                             (np.random.randn(64, 49, 128),)],
                                     column_names=["col"])
else:
    raise RuntimeError('datatype not in [video, image]')


class Network(nn.Cell):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        init_tensor = np.random.randn(
            (2 * self.window_size[0] - 1)
            * (2 * self.window_size[1] - 1)
            * (2 * self.window_size[2] - 1),
            self.num_heads
        )
        init_tensor = Tensor(init_tensor, dtype=ms.float16)
        self.relative_position_bias_table = Parameter(init_tensor, requires_grad=True)

        coords_d = np.arange(self.window_size[0])
        coords_h = np.arange(self.window_size[1])
        coords_w = np.arange(self.window_size[2])
        coords = np.stack(np.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))  # 3, Wd, Wh, Ww
        coords_flatten = coords.reshape(3, coords.shape[1] * coords.shape[2] * coords.shape[3])  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = np.transpose(relative_coords, (1, 2, 0))  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        self.relative_position_index = Tensor(relative_coords.sum(-1), dtype=ms.int32)  # Wd*Wh*Ww, Wd*Wh*Ww

    def construct(self, x):
        B_, N, C = x.shape

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index[:N, :N].reshape(-1)]
        out = relative_position_bias.sum() / relative_position_bias.size
        return out.abs()

model = Network(window_size=[8, 7, 7], num_heads=4)
optimizer = nn.AdamWeightDecay(
    model.trainable_params(),
    learning_rate=0.0001,
)
# profiler = Profiler(output_path = '/home/ma-user/work/summary_dir/profiler_data_debug')
trainer = Model(model, optimizer=optimizer)
steps_per_epoch = train_dataset.get_dataset_size()
callback = [LossMonitorMAE(log=logger), ]
trainer.train(5, train_dataset, callbacks=callback, dataset_sink_mode=False)
# profiler.analyse()
