import numpy as np
import torch.nn as nn
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def calculate_iou(segs, labels, eps=1e-8):
    intersaction = segs * labels
    iou = (eps+intersaction.sum())/(segs.sum()+labels.sum()-intersaction.sum() + eps)
    return iou

def calculate_pixel_f1(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    return f1, precision, recall

def calculate_batch_score(segs, labels):
    batch_size = segs.shape[0]
    batch_f1, batch_iou = 0.0, 0.0
    for i in range(batch_size):
        pd = segs[i]
        gt = labels[i]
        fake_seg = pd.detach().cpu().numpy()
        fake_gt = gt.detach().cpu().numpy()
        fake_seg = np.where(fake_seg<0.5, 0.0, 1.0)
        # print(fake_seg.shape, fake_gt.shape)
        f1, p, r = calculate_pixel_f1(fake_seg.flatten(),fake_gt.flatten())
        batch_f1 += f1
        iou = calculate_iou(fake_seg.flatten(),fake_gt.flatten())
        batch_iou += iou
    return batch_f1 / batch_size, batch_iou / batch_size

def score_cls(pred, labels):
    pred_untampered_index = np.argwhere(labels==0)
    pred_tampered_index = np.argwhere(labels==1)
    pred_tampered = pred[pred_tampered_index]
    pred_untampered = pred[pred_untampered_index]
    thres = np.percentile(pred_untampered, np.arange(90,100,1))
    recall = np.mean(np.greater(pred_tampered[:, np.newaxis], thres).mean(axis=0))
    return recall * 100


def get_threshold(preds, untampers):
    # preds = [np.max(cv2.imread(os.path.join(submission_folder, 'submission', path), cv2.IMREAD_GRAYSCALE))
    #         for path in untampers[:, 0]]
    # preds = preds[untampers]
    # for untamper in untampers:
    #     print(untamper)
    _preds = [np.max(preds[untamper[0]]) for untamper in untampers]
    return np.percentile(_preds, np.arange(91, 100, 1))

def subprocess(preds, masks, thres, item):
    # imgpath = os.path.join(submission_folder, 'submission', item)
    # maskpath = os.path.join('competition2/ground_truth', item)
    # pred = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    # mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    pred = preds[item]
    mask = masks[item]
    iou_score = iou(pred, mask, thres)
    return iou_score

def iou(pred, mask, thres, eps=1e-8):
    mask = np.clip(mask, 0, 1)
    iou_m = []

    first_threshold = int(thres[0])
    num_thresholds = 256 - first_threshold
    iou_map = np.zeros(num_thresholds)
    for i in range(num_thresholds):
        tmp = np.zeros_like(pred)
        thre = i + first_threshold
        tmp[pred > thre] = 1
        iou_value = (np.count_nonzero(np.logical_and(
            mask, tmp)) +eps)/ (np.count_nonzero(np.logical_or(mask, tmp)) +eps)
        iou_map[i] = iou_value 
    for item in thres:
        iou_m.append(np.max(iou_map[int(item)-first_threshold:]))
    # print(iou_m)
    return np.sum(iou_m)

def score_seg(pred, masks):
    labels = []
    for mask in masks:
        if mask.max()==1:
            labels.append(1)
        else:
            labels.append(0)
    labels = np.array(labels)
    pred_untampered_index = np.argwhere(labels==0)
    pred_tampered_index = np.argwhere(labels==1)

    thres = get_threshold(np.uint8(pred), pred_untampered_index)
    # print(thres)
    # thres = [127, 128, 129, 130, 131, 132, 133, 134, 135]
    # thres = [0, 1, 2, 3, 4, 5, 6, 7, 8]

    num_tampers = pred_tampered_index.shape[0]
    scores = np.empty(num_tampers)
    # print(num_tampers)

    # with ProcessPoolExecutor() as ex:
    #     future_scores = [ex.submit(subprocess, pred, masks, thres, item) for item in pred_tampered_index]
    #     for i, future in enumerate(as_completed(future_scores)):
    #         scores[i] = future.result()

    for i, index in enumerate(tqdm(pred_tampered_index)):
        scores[i] = subprocess(np.uint8(pred), masks, np.uint8(thres), index)

    return np.mean(scores)



class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


def get_one_hot(label, N):
    label = label.long()
    size = list(label.size())
    size[1] = N
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N).to(label.device)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    return ones.view(*size)