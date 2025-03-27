#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
from batchgenerators.augmentations.utils import pad_nd_image #于对 n 维图像进行填充，以达到所需的大小
from nnunet.utilities.random_stuff import no_op # 用于执行不操作的函数
from nnunet.utilities.to_torch import to_cuda, maybe_to_torch #to_cuda 和 maybe_to_torch：这些函数用于将数据转换为 PyTorch 可以使用的格式，并将数据移动到 GPU 设备（如果存在）上
from torch import nn
import torch
from scipy.ndimage.filters import gaussian_filter #用于对 n 维数组应用高斯滤波
from typing import Union, Tuple, List

from torch.cuda.amp import autocast#用于在 PyTorch GPU 计算上自动启用更快的混合精度模式。混合精度模式可以提高模型训练速度，并且通常不会对模型精度造成显着影响


class NeuralNetwork(nn.Module):
    def __init__(self): # __init__：构造函数，调用了父类的构造函数
        super(NeuralNetwork, self).__init__()

    def get_device(self): # get_device：用于获取神经网络当前运行的设备（CPU 或 GPU）
        if next(self.parameters()).device.type == "cpu":
            return "cpu"
        else:
            return next(self.parameters()).device.index

    def set_device(self, device): #于将神经网络运行到指定的设备上
        if device == "cpu":
            self.cpu()
        else:
            self.cuda(device)

    def forward(self, x): #用于定义神经网络的前向传播过程。但是，在这个类中并没有实现该方法，需要由其子类实现
        raise NotImplementedError


class SegmentationNetwork(NeuralNetwork):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # if we have 5 pooling then our patch size must be divisible by 2**5
        #用于存储神经网络的输入形状必须可以被什么数字整除。如果我们有5个池化层，那么输入形状必须被2^5整除。例如，在一个2d的网络中，我们对x维度做5次池化，对y做6次池化，那么输入形状应为（2^5,2^6）
        self.input_shape_must_be_divisible_by = None  # for example in a 2d network that does 5 pool in x and 6 pool
        # in y this would be (32, 64)

        # we need to know this because we need to know if we are a 2d or a 3d netowrk
        self.conv_op = None  # nn.Conv2d or nn.Conv3d #存储了使用的卷积操作，可以是 nn.Conv2d 或 nn.Conv3d

        # this tells us how many channels we have in the output. Important for preallocation in inference
        self.num_classes = None  # number of channels in the output #存储输出通道数量，在推理时用于预先分配内存。

        # depending on the loss, we do not hard code a nonlinearity into the architecture. To aggregate predictions
        # during inference, we need to apply the nonlinearity, however. So it is important to let the newtork know what
        # to apply in inference. For the most part this will be softmax
        self.inference_apply_nonlin = lambda x: x  # softmax_helper #存储推理时应用的非线性函数，通常是 softmax

        # This is for saving a gaussian importance map for inference. It weights voxels higher that are closer to the
        # center. Prediction at the borders are often less accurate and are thus downweighted. Creating these Gaussians
        # can be expensive, so it makes sense to save and reuse them.
        self._gaussian_3d = self._patch_size_for_gaussian_3d = None #_gaussian_3d用于存储 3D 高斯重要图，以用于推理, _patch_size_for_gaussian_3d储用于生成 3D 高斯重要图的小批量大小
        self._gaussian_2d = self._patch_size_for_gaussian_2d = None

    '''
    使用这个函数来预测3D图像。不管网络是2D还是3D U-Net，它都会自动检测到并运行适当的代码。在运行预测时，您需要指定是否要运行完全卷积或基于滑动窗口的推理。我们非常强烈建议您使用带有默认设置的滑动窗口。用户有责任确保网络处于适当的模式(评估以进行推理!)。如果网络不处于eval模式，它将打印警告。函数各参数含义：
    x: 输入数据，必须是形状为（c，x，y，z）的 nd.ndarray。
    do_mirroring: 如果为 True，则在测试时使用数据增强（镜像）
    mirror_axes: 确定用于镜像的轴。默认情况下，将沿着三个轴进行镜像。
    use_sliding_window: 如果为 True，则运行滑动窗口预测。强烈推荐！这也是默认值
    step_size:当使用滑动窗口预测时，该参数决定相邻预测之间的举例，该参数越小，预测越密集，0.5是默认值，step_size不能大于1
    use_gaussian: 仅适用于滑动窗口预测。如果设为 True，则使用高斯重要性加权来加权更靠近当前补丁中心的预测，而不是在边界处的预测。这是因为分割精度向边界减小。默认（和推荐）值：True。
    pad_border_mode: 边界填充模式.指的是用于填充图像边界的方法。这种填充方法用于控制在图像进行尺寸变换时（例如缩放）如何处理图像边缘的像素。一些常见的填充模式包括"constant"（使用常数填充）、"reflect"（使用图像边界的反射）和"replicate"（复制图像边界的像素）。
    pad_kwargs: 请不要改动。 边界填充的参数
    all_in_gpu: 实验性质。您可能希望保留原样。
    verbose: 是否需要文字输出？如果是，请将其设为 True。
    mixed_precision: 如果设为 True，则将使用 autocast() 进行混合精度推理。
    返回值为Tuple类型，包含两个np.ndarray类型的数组。
    '''
    def predict_3D(self, x: np.ndarray, do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                   use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 3D image. It does not matter whether the network is a 2D or 3D U-Net, it will
        detect that automatically and run the appropriate code.

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y, z).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :param mixed_precision: if True, will run inference in mixed precision with autocast()
        :return:
        """
        torch.cuda.empty_cache() #PyTorch 中的一个函数，用于清除 CUDA 缓存。它可以帮助释放 GPU 内存，从而提高 GPU 的利用效率。

        assert step_size <= 1, 'step_size must be smaller than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        # if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes) #verbose 是一个布尔类型的变量，用于表示是否需要输出详细信息

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention # 此代码检查镜像轴是否正确。如果conv_op为nn.Conv2d，则mirror_axes的最大值必须小于等于1。如果conv_op= nn.Conv3d，则mirror_axes的最大值必须小于等于2。否则，将引发ValueError，消息为"mirror axes. duh"
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv3d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu,
                                                                     verbose=verbose)
                    else:
                        res = self._internal_predict_3D_3Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                # 这段代码用于判断是否使用滑动窗口预测，如果使用，则调用self._internal_predict_3D_3Dconv_tiled 方法，否则调用
                # self._internal_predict_3D_3Dconv 方法。参数分别是输入数据、步长、是否使用镜像、镜像轴、补丁大小、类别顺序、是否使用高斯、边界模式、参数字典、											  #是否在GPU上运行以及是否输出详细信息。
                elif self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_2Dconv_tiled(x, patch_size, do_mirroring, mirror_axes, step_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, False)
                    else:
                        res = self._internal_predict_3D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    def predict_2D(self, x, do_mirroring: bool, mirror_axes: tuple = (0, 1, 2), use_sliding_window: bool = False,
                   step_size: float = 0.5, patch_size: tuple = None, regions_class_order: tuple = None,
                   use_gaussian: bool = False, pad_border_mode: str = "constant",
                   pad_kwargs: dict = None, all_in_gpu: bool = False,
                   verbose: bool = True, mixed_precision: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use this function to predict a 2D image. If this is a 3D U-Net it will crash because you cannot predict a 2D
        image with that (you dummy).

        When running predictions, you need to specify whether you want to run fully convolutional of sliding window
        based inference. We very strongly recommend you use sliding window with the default settings.

        It is the responsibility of the user to make sure the network is in the proper mode (eval for inference!). If
        the network is not in eval mode it will print a warning.

        :param x: Your input data. Must be a nd.ndarray of shape (c, x, y).
        :param do_mirroring: If True, use test time data augmentation in the form of mirroring
        :param mirror_axes: Determines which axes to use for mirroing. Per default, mirroring is done along all three
        axes
        :param use_sliding_window: if True, run sliding window prediction. Heavily recommended! This is also the default
        :param step_size: When running sliding window prediction, the step size determines the distance between adjacent
        predictions. The smaller the step size, the denser the predictions (and the longer it takes!). Step size is given
        as a fraction of the patch_size. 0.5 is the default and means that wen advance by patch_size * 0.5 between
        predictions. step_size cannot be larger than 1!
        :param patch_size: The patch size that was used for training the network. Do not use different patch sizes here,
        this will either crash or give potentially less accurate segmentations
        :param regions_class_order: Fabian only
        :param use_gaussian: (Only applies to sliding window prediction) If True, uses a Gaussian importance weighting
         to weigh predictions closer to the center of the current patch higher than those at the borders. The reason
         behind this is that the segmentation accuracy decreases towards the borders. Default (and recommended): True
        :param pad_border_mode: leave this alone
        :param pad_kwargs: leave this alone
        :param all_in_gpu: experimental. You probably want to leave this as is it
        :param verbose: Do you want a wall of text? If yes then set this to True
        :return:
        """
        torch.cuda.empty_cache()

        assert step_size <= 1, 'step_size must be smaler than 1. Otherwise there will be a gap between consecutive ' \
                               'predictions'

        if self.conv_op == nn.Conv3d:
            raise RuntimeError("Cannot predict 2d if the network is 3d. Dummy.")

        # if verbose: print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3) for a 2d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if max(mirror_axes) > 1:
                raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 3, "data must have shape (c,x,y)"

        if mixed_precision:
            context = autocast
        else:
            context = no_op

        with context():
            with torch.no_grad():
                if self.conv_op == nn.Conv2d:
                    if use_sliding_window:
                        res = self._internal_predict_2D_2Dconv_tiled(x, step_size, do_mirroring, mirror_axes, patch_size,
                                                                     regions_class_order, use_gaussian, pad_border_mode,
                                                                     pad_kwargs, all_in_gpu, verbose)
                    else:
                        res = self._internal_predict_2D_2Dconv(x, patch_size, do_mirroring, mirror_axes, regions_class_order,
                                                               pad_border_mode, pad_kwargs, verbose)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")

        return res

    '''
    这是一个静态方法，它返回一个高斯函数数组。输入参数 patch_size 表示高斯数组的尺寸，sigma_scale(高斯分布的标准差的大小) 默认值为1/8。在函数内部，首先定义一个零值数组 tmp，
    然后计算数组的中心坐标 center_coords，并且计算标准差 sigmas。接下来，将数组 tmp 的中心赋值为 1，并使用高斯滤波函数计算出高斯重要性图 gaussian_importance_map。
    最后，将 gaussian_importance_map 转换为 float32 类型，如果 gaussian_importance_map 为 0，则把它改为最小值，以防止出现 NaN 值，最后返回高斯数组
    '''
    @staticmethod
    def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
        tmp = np.zeros(patch_size)
        center_coords = [i // 2 for i in patch_size]
        sigmas = [i * sigma_scale for i in patch_size]
        tmp[tuple(center_coords)] = 1
        gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
        gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
        gaussian_importance_map = gaussian_importance_map.astype(np.float32)

        # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
        gaussian_importance_map[gaussian_importance_map == 0] = np.min(
            gaussian_importance_map[gaussian_importance_map != 0])

        return gaussian_importance_map

    '''
    这是一个计算滑动窗口步数的静态方法。它需要传入patch_size，image_size，和step_size三个参数，分别表示patch的大小，图像的大小和滑动步长。
    该方法主要实现了以下功能：
    判断图像大小是否大于等于patch大小。
    判断step_size是否在0到1之间。
    计算出每个维度上的步长，以达到将整个图像滑动的目的。
    最后，该方法返回一个包含每个维度步长的列表。
    '''
    @staticmethod
    def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...], image_size: Tuple[int, ...], step_size: float) -> List[List[int]]:
        assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
        #'''这个assert语句检查image_size和patch_size两个元组的每一个元素，如果image_size的每一个元素都大于等于patch_size的对应元素，那么就不会抛出错误；否则，如果image_size的任意一个元素小于patch_size的对应元素，就会抛出错误''
        assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'

        # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
        # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
        target_step_sizes_in_voxels = [i * step_size for i in patch_size] ## 步长最多为patch_size * step_size，但可能会更窄。例如，如果我们有图像大         #小110，patch_size=64和步长0.5，那么我们希望从坐标0，23，46开始进行3步。

        num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

        #'''这里计算了每个维度需要多少步来完整地覆盖图像。首先，为了覆盖整个图像，每个步骤的大小最多为patch_size * step_size。
        # 其次，如果步长不足以覆盖整个图像，则应扩大步长，以确保图像被完全覆盖。因此，对于每个维度，实际步长是从图像大小减去patch_size除以步数，并将其向上取整加1。'''
        steps = []
        for dim in range(len(patch_size)):
            # the highest step value for this dimension is
            max_step_value = image_size[dim] - patch_size[dim]
            if num_steps[dim] > 1:
                actual_step_size = max_step_value / (num_steps[dim] - 1)
            else:
                actual_step_size = 99999999999  # does not matter because there is only one step at 0

            steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

            steps.append(steps_here)

        return steps

    #方法用于使用3D卷积来对3D数据进行分割预测。它的具体实现方式是：它通过将大型3D输入图像拆分为小型3D子图像，在每个子图像上应用3D卷积，然后在最终的预测上合并输出来实现功能
    def _internal_predict_3D_3Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        '''对输入的数据 "x" 进行补全，以达到满足窗口滑动需求的数据大小，在使用滑动窗口进行推理时，图像的大小至少要与patch大小相同，不管该形状
        是否可以被2的num_pool次方整除，只要patch大小是整除即可。这个函数返回两个结果，一个是被补全后的数据 "data"，另一个是用于滑动窗口的索引信息 "slicer"。'''
        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        '''steps是窗口滑动步数的列表，其计算方法是通过调用私有方法_compute_steps_for_sliding_window来获得。num_tiles是所有步数的乘积，即表示总共需要切分的图像数量。'''
        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
                if verbose: print("done")
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

            '''若将 non_blocking 设置为 True，那么 GPU 上的张量转移将会是非阻塞性的，也就是说，该代码不会等到 GPU 上的张量转移完成后再继续执行，
            而是直接继续执行。若将 non_blocking 设置为 False，则该代码会等待 GPU 上的张量转移完成后再继续执行。'''
            #predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)


        else:
            gaussian_importance_map = None

        if all_in_gpu:#如果推理过程只在gpu上进行
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min() #'''将变量gaussian_importance_map中值为0的位置的值替换为gaussian_importance_map中非0位置最小值'''

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = self._internal_maybe_mirror_and_pred_3D(
                        data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], mirror_axes, do_mirroring,
                        gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        aggregated_results /= aggregated_nb_of_predictions
        del aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = aggregated_results.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = aggregated_results.detach().cpu().numpy()
            else:
                class_probabilities_here = aggregated_results
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            aggregated_results = aggregated_results.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, aggregated_results

    '''
    预测2D分割结果的函数。它的输入是一个三维的数组（即输入的图像）和一些预测的参数，如最小的图像大小，是否需要镜像等。
首先，使用 pad_nd_image 函数对输入的图像进行填充，以保证图像的大小符合预定要求。然后调用 _internal_maybe_mirror_and_pred_2D 函数，得到预测的概率分布。最后，将概率分布切片以得到最终的分割结果。
如果没有指定 regions_class_order 参数，那么将概率分布中最大值所在的索引作为分割结果。否则，按照 regions_class_order 的顺序，对于每一个类别，将概率大于 0.5 的位置标记为该类别的编号。最后，将分割结果和概率分布返回。
    '''
    def _internal_predict_2D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 3, "x must be (c, x, y)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_2D_2Dconv'
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D(data[None], mirror_axes, do_mirroring,
                                                                          None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_predict_3D_3Dconv(self, x: np.ndarray, min_size: Tuple[int, ...], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1, 2), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_3D_3Dconv'
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_3D(data[None], mirror_axes, do_mirroring,
                                                                          None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_maybe_mirror_and_pred_3D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        x = maybe_to_torch(x)
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]),
                                   dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, ))))
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_maybe_mirror_and_pred_2D(self, x: Union[np.ndarray, torch.tensor], mirror_axes: tuple,
                                           do_mirroring: bool = True,
                                           mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = maybe_to_torch(x)
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, ))))
                result_torch += 1 / num_results * torch.flip(pred, (3, ))

            if m == 2 and (0 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2, ))))
                result_torch += 1 / num_results * torch.flip(pred, (2, ))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_predict_2D_2Dconv_tiled(self, x: np.ndarray, step_size: float, do_mirroring: bool, mirror_axes: tuple,
                                          patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
                                          pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool,
                                          verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_2d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose: print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                             device=self.get_device())

            if verbose: print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose: print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half,
                                                       device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D(
                    data[None, :, lb_x:ub_x, lb_y:ub_y], mirror_axes, do_mirroring,
                    gaussian_importance_map)[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    '''
    它在3D图像分割任务中实现预测，并使用二维卷积来实现这一目标。该方法首先将输入数据 x 转换为 PyTorch 张量，并创建一个结果张量 result_torch。
    然后，该方法使用两个循环：第一个循环处理每一个 Z 轴的片段，第二个循环处理输入图像上的每一个 2D 图像。在每个 2D 图像上，该方法使用网络 self 
    预测该图像的输出，并将结果累加到结果张量 result_torch 中。最后，该方法返回结果张量。
    '''
    def _internal_predict_3D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                    mirror_axes: tuple = (0, 1), regions_class_order: tuple = None,
                                    pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                    all_in_gpu: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv(
                x[:, s], min_size, do_mirroring, mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    '''
    使用了伪3D卷积（Pseudo 3D Convolution）的思想。它的输入是一个四维数组，表示一个3D图像，每一维分别表示通道数、深度（Z轴）、高（Y轴）、宽（X轴）。
首先，它在图像的深度方向上做了一些扩展，使用0填充，以构造伪3D的数据。然后，它在深度方向上按照固定的步长进行循环，每次循环取出一个具有一定长度的数据块，将其矩阵拉成2D，
然后送入另一个内部函数_internal_predict_2D_2Dconv进行预测。最后，将各个预测结果连接在一起，并返回分割结果和对应的softmax预测结果。
伪3D卷积的目的是通过利用卷积操作对图像数据进行特征提取，同时在保持3D图像的空间结构的情况下，减少计算量。
    '''
    def predict_3D_pseudo3D_2Dconv(self, x: np.ndarray, min_size: Tuple[int, int], do_mirroring: bool,
                                   mirror_axes: tuple = (0, 1), regions_class_order: tuple = None,
                                   pseudo3D_slices: int = 5, all_in_gpu: bool = False,
                                   pad_border_mode: str = "constant", pad_kwargs: dict = None,
                                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        assert pseudo3D_slices % 2 == 1, "pseudo3D_slices must be odd"
        extra_slices = (pseudo3D_slices - 1) // 2

        shp_for_pad = np.array(x.shape)
        shp_for_pad[1] = extra_slices

        pad = np.zeros(shp_for_pad, dtype=np.float32)
        data = np.concatenate((pad, x, pad), 1)

        predicted_segmentation = []
        softmax_pred = []
        for s in range(extra_slices, data.shape[1] - extra_slices):
            d = data[:, (s - extra_slices):(s + extra_slices + 1)]
            d = d.reshape((-1, d.shape[-2], d.shape[-1]))
            pred_seg, softmax_pres = \
                self._internal_predict_2D_2Dconv(d, min_size, do_mirroring, mirror_axes,
                                                 regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

    '''
    该方法接收一个四维的numpy数组 x，其中的前三维为图像的通道数、长度、宽度，最后一维为图像的堆叠数量。该方法通过循环遍历每一个堆叠的2D图像，
    并调用另一个私有方法 _internal_predict_2D_2Dconv 进行分割预测，最后将所有2D图像的预测结果拼接起来，返回分割预测结果和置信度预测结果。
    '''
    def _internal_predict_3D_2Dconv_tiled(self, x: np.ndarray, patch_size: Tuple[int, int], do_mirroring: bool,
                                          mirror_axes: tuple = (0, 1), step_size: float = 0.5,
                                          regions_class_order: tuple = None, use_gaussian: bool = False,
                                          pad_border_mode: str = "edge", pad_kwargs: dict =None,
                                          all_in_gpu: bool = False,
                                          verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(x.shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled(
                x[:, s], step_size, do_mirroring, mirror_axes, patch_size, regions_class_order, use_gaussian,
                pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred


if __name__ == '__main__':
    print(SegmentationNetwork._compute_steps_for_sliding_window((30, 224, 224), (162, 529, 529), 0.5))
    print(SegmentationNetwork._compute_steps_for_sliding_window((30, 224, 224), (162, 529, 529), 1))
    print(SegmentationNetwork._compute_steps_for_sliding_window((30, 224, 224), (162, 529, 529), 0.1))

    print(SegmentationNetwork._compute_steps_for_sliding_window((30, 224, 224), (60, 448, 224), 1))
    print(SegmentationNetwork._compute_steps_for_sliding_window((30, 224, 224), (60, 448, 224), 0.5))

    print(SegmentationNetwork._compute_steps_for_sliding_window((30, 224, 224), (30, 224, 224), 1))
    print(SegmentationNetwork._compute_steps_for_sliding_window((30, 224, 224), (30, 224, 224), 0.125))


    print(SegmentationNetwork._compute_steps_for_sliding_window((123, 54, 123), (246, 162, 369), 0.25))



'''
总结：

predict_3D
_internal_predict_3D_3Dconv_tiled该方法通过将输入体积划分为多个重叠的tiles，并在每个tiles上运行3D卷积来实现预测功能。每个tiles的预测结果然后被合并以形成整个体积的最终预测。此分块策略用于处理可能不适合存储器的大型输入体积。
_internal_predict_3D_3Dconv使用3D卷积来对3D数据进行分割预测
_internal_maybe_mirror_and_pred_3D对原始输入图像进行镜像操作（如果需要），然后使用三维卷积神经网络（3D CNNs）对图像进行预测，得到对整个图像的最终预测结果。对于每一个镜像方式，它都会对输入的三维图像进行预测，并将预测结果累加在一起，最后得到的结果是所有镜像方式的预测结果的平均值,参数mult是对预测结果的乘数,用来调整结果的大小。
_internal_predict_3D_2Dconv它在3D图像分割任务中实现预测，并使用二维卷积来实现这一目标。该方法首先将输入数据 x 转换为 PyTorch 张量，并创建一个结果张量 result_torch。然后，该方法使用两个循环：第一个循环处理每一个 Z 轴的片段，第二个循环处理输入图像上的每一个 2D 图像。在每个 2D 图像上，该方法使用网络 self 预测该图像的输出，并将结果累加到结果张量 result_torch 中。最后，该方法返回结果张量。
predict_3D_pseudo3D_2Dconv使用3D的数据做分割预测，但是通过在2D卷积中对每一层进行预测，并将所有层的结果进行累加得到最终预测结果。方法的具体实现可以在代码中查看。
_internal_predict_3D_2Dconv_tiled将3D图像切片成若干个2D图像，然后对每个2D图像分别进行预测，最后将预测结果合并起来得到最终结果。

predict_2D
_internal_predict_2D_2Dconv对每个patch运行2D卷积，并将每个patch的预测结果连接起来，以形成整个图像的最终预测结果，从而实现预测功能。这种基于patch的策略通常用于处理可能不适合内存的大图像。
_internal_maybe_mirror_and_pred_2D
_internal_predict_2D_2Dconv_tiled对2维的输入进行分块预测。它首先将输入分成多个小的块，再分别对每个小块使用2维卷积神经网络进行预测，最后将预测结果拼接在一起。

_get_gaussian网络架构的 CNN 层中创建高斯核的函数。此函数用于创建具有权重的张量，以便作为卷积核用于训练过程。具体来说，它创建一个形状为 [width, height, in_channels, out_channels] 的 4 维变量，其中 width 和 height 为卷积核的大小，in_channels 和 out_channels 分别表示输入和输出特征图的通道数。每个元素都可以使用正态分布函数计算出来，以随机初始化核中的权重

_compute_steps_for_sliding_window计算步骤的函数，用于利用滑动窗口技术对图像的批量处理。这个函数能够根据给定的滑动窗口大小，确定滑动窗口的步数，从而使得给定的图像能够被很好地覆盖。  
'''