import torch

from segment.segment_network.model_restore import load_model_and_checkpoint_files
import copy
import numpy as np
from typing import Optional
from skimage.metrics import structural_similarity

def get_seg_result(d, seg_trainer, target_label, do_tta=True, all_in_gpu=True, step_size=0.5,
                   mixed_precision=True):
    softmax = seg_trainer['fold0'].predict_preprocessed_data_return_seg_and_softmax(
        d, do_mirroring=do_tta, mirror_axes=seg_trainer['fold0'].data_aug_params['mirror_axes'], use_sliding_window=True,
        step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
        mixed_precision=mixed_precision)[1]

    softmax += seg_trainer['fold1'].predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=seg_trainer['fold1'].data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]

    softmax += seg_trainer['fold2'].predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=seg_trainer['fold2'].data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]

    softmax += seg_trainer['fold3'].predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=seg_trainer['fold3'].data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]

    softmax += seg_trainer['fold4'].predict_preprocessed_data_return_seg_and_softmax(
            d, do_mirroring=do_tta, mirror_axes=seg_trainer['fold4'].data_aug_params['mirror_axes'], use_sliding_window=True,
            step_size=step_size, use_gaussian=True, all_in_gpu=all_in_gpu,
            mixed_precision=mixed_precision)[1]

    softmax /= len(seg_trainer)

    out_int64 = softmax.argmax(1).cpu().detach().numpy()
    out_predict = out_int64.astype(np.uint8)
    get_dice = dice(out_predict, target_label)

    return torch.tensor(get_dice, dtype=torch.float32).to('cuda')


def load_5segmodels():
    seg_model_checkpoint = '/home/ruru/Documents/work/ACDC/acdc-testing/nnUnet_pretraind_model/Task027_ACDC/2d/Task027_ACDC/nnUNetTrainerV2__nnUNetPlansv2.1'
    folds = None
    mixed_precision = True
    seg_checkpoint_name = 'model_final_checkpoint'
    seg_trainer, seg_params = load_model_and_checkpoint_files(seg_model_checkpoint, folds, mixed_precision=mixed_precision, checkpoint_name=seg_checkpoint_name)
    fold0_model = copy.deepcopy(seg_trainer)
    fold0_model.load_checkpoint_ram(seg_params[0], False)

    fold1_model = copy.deepcopy(seg_trainer)
    fold1_model.load_checkpoint_ram(seg_params[1], False)

    fold2_model = copy.deepcopy(seg_trainer)
    fold2_model.load_checkpoint_ram(seg_params[2], False)

    fold3_model = copy.deepcopy(seg_trainer)
    fold3_model.load_checkpoint_ram(seg_params[3], False)

    fold4_model = copy.deepcopy(seg_trainer)
    fold4_model.load_checkpoint_ram(seg_params[4], False)

    models_dict = {'fold0': fold0_model,
           'fold1': fold1_model,
           'fold2': fold2_model,
           'fold3': fold3_model,
           'fold4': fold4_model
           }
    return models_dict



def ssim(
    gt: np.ndarray, pred: np.ndarray, maxval: Optional[float] = None
) -> np.ndarray:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == 3:
        raise ValueError("Unexpected number of dimensions in ground truth.")
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() if maxval is None else maxval

    ssim = np.array([0])
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    return ssim / gt.shape[0]



def assert_shape(test, reference):

    assert test.shape == reference.shape, "Shape mismatch: {} and {}".format(
        test.shape, reference.shape)
class ConfusionMatrix:

    def __init__(self, test=None, reference=None):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.reference_empty = None
        self.reference_full = None
        self.test_empty = None
        self.test_full = None
        self.set_reference(reference)
        self.set_test(test)

    def set_test(self, test):

        self.test = test
        self.reset()

    def set_reference(self, reference):

        self.reference = reference
        self.reset()

    def reset(self):

        self.tp = None
        self.fp = None
        self.tn = None
        self.fn = None
        self.size = None
        self.test_empty = None
        self.test_full = None
        self.reference_empty = None
        self.reference_full = None

    def compute(self):

        if self.test is None or self.reference is None:
            raise ValueError("'test' and 'reference' must both be set to compute confusion matrix.")

        assert_shape(self.test, self.reference)

        self.tp = ((self.test != 0) * (self.reference != 0)).sum(-1).sum(-1)
        self.fp = ((self.test != 0) * (self.reference == 0)).sum(-1).sum(-1)
        self.tn = ((self.test == 0) * (self.reference == 0)).sum(-1).sum(-1)
        self.fn = ((self.test == 0) * (self.reference != 0)).sum(-1).sum(-1)
        self.size = int(np.prod(self.reference.shape, dtype=np.int64))
        self.test_empty = not np.any(self.test)
        self.test_full = np.all(self.test)
        self.reference_empty = not np.any(self.reference)
        self.reference_full = np.all(self.reference)

    def get_matrix(self):

        for entry in (self.tp, self.fp, self.tn, self.fn):
            if entry is None:
                self.compute()
                break

        return self.tp, self.fp, self.tn, self.fn

    def get_size(self):

        if self.size is None:
            self.compute()
        return self.size

    def get_existence(self):

        for case in (self.test_empty, self.test_full, self.reference_empty, self.reference_full):
            if case is None:
                self.compute()
                break

        return self.test_empty, self.test_full, self.reference_empty, self.reference_full


def dice(test=None, reference=None, confusion_matrix=None, nan_for_nonexisting=True, **kwargs):
    """2TP / (2TP + FP + FN)"""

    if confusion_matrix is None:
        confusion_matrix = ConfusionMatrix(test, reference)

    tp, fp, tn, fn = confusion_matrix.get_matrix()
    test_empty, test_full, reference_empty, reference_full = confusion_matrix.get_existence()

    if test_empty and reference_empty:
        if nan_for_nonexisting:
            return float("NaN")
        else:
            return 0.

    return 2. * tp / (2 * tp + fp + fn)