from torch.autograd import Function
import torch


class DiceLoss(Function):
    '''
    Compute energy based on dice coefficient.
    Aims to maximize dice coefficient.
    '''
    def __init__(self, *args, **kwargs):
        pass

    def forward(self, input, target, save=True):
        if save:
            self.save_for_backward(input, target)
        eps = 0.00001
        _, result_ = input.max(1)
        result_ = torch.squeeze(result_)
        if input.is_cuda:
            result = torch.cuda.FloatTensor(result_.size())
            self.target_ = torch.cuda.FloatTensor(target.size())
        else:
            result = torch.FloatTensor(result_.size())
            self.target_ = torch.FloatTensor(target.size())
        result.copy_(result_)
        self.target_.copy_(target)
        target = self.target_
#       print(input)
        intersect = torch.dot(result, target)
        # binary values so sum the same as sum of squares
        result_sum = torch.sum(result)
        target_sum = torch.sum(target)
        union = result_sum + target_sum

        # the target volume can be empty - so we still want to
        # end up with a score of 1 if the result is 0/0
        dice = 2*intersect / (union + eps)
        print('union: {:.3f}\t intersect: {:.6f}\t target_sum: {:.0f} pred_sum: {:.0f} dice_coefficient: {:.7f}'.format(
            union, intersect, target_sum, result_sum, dice))
        out = torch.FloatTensor(1).fill_(dice)
        if input.is_cuda:
            out = out.cuda() # added by Chao.
        self.intersect, self.union = intersect, union
        return out

    def backward(self, grad_output):
        input, _ = self.saved_tensors
        intersect, union = self.intersect, self.union
        target = self.target_
        part1 = torch.div(target, union)
        part2_2 = intersect/(union*union)
        part2 = torch.mul(input[:, 1], part2_2)
        dDice = torch.add(torch.mul(part1, 2), torch.mul(part2, -4))
        grad_input = torch.cat((torch.mul(dDice, grad_output[0]).view(-1,1), torch.mul(dDice, -grad_output[0]).view(-1,1)), 1)
        return grad_input, None


def dice_loss(input, target):
    return DiceLoss()(input, target)


def dice_error(input, target):
    eps = 0.00001
    result = torch.cuda.FloatTensor(input)
    target = torch.cuda.FloatTensor(target)

    intersect = result * target
    result_sum = torch.sum(result)
    target_sum = torch.sum(target)
    union = result_sum + target_sum
    # print(f'union:', union)
    intersect = torch.sum(intersect)
    dice = 2*intersect / (union + eps)
    print(f'dice:', dice)
    return dice


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TP : True Positive
    # FN : False Negative
    TP = (SR == 1) & (GT == 1)
    FN = (SR == 0) & (GT == 1)
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    return SE


def get_specificity(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TN : True Negative
    # FP : False Positive
    TN = (SR == 0) & (GT == 0)
    FP = (SR == 1) & (GT == 0)
    SP = float(torch.sum(TN)) / (float(torch.sum(TN + FP)) + 1e-6)
    return SP


def get_precision(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TP : True Positive
    # FP : False Positive
    TP = (SR == 1) & (GT == 1)
    FP = (SR == 1) & (GT == 0)
    PC = float(torch.sum(TP)) / (float(torch.sum(TP + FP)) + 1e-6)
    return PC


def get_DC(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == torch.max(GT)
    uion = SR * GT
    Inter = torch.sum(uion)
    DC = float(2 * Inter) / (float(torch.sum(SR) + torch.sum(GT)) + 1e-6)
    return DC


def calVOE(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    uion = SR * GT
    Inter = torch.sum(uion)
    union = torch.sum(GT | SR)
    VOE = 1 - float(Inter) / (float(union) + 1e-6)
    return VOE


def calRVD(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    RVD_s = torch.sum(GT).item()
    RVD_t = torch.sum(SR).item()
    RVD = RVD_t/RVD_s - 1
    return RVD