import torch
import torch.nn.functional as F

def laplacian_edge(img):
    laplacian_filter = torch.Tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filter = torch.reshape(laplacian_filter, [1, 1, 3, 3])
    filter = filter.cuda()
    lap_edge = F.conv2d(img, filter, stride=1, padding=1)
    return lap_edge

def gradient_x(img):
    sobel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    filter = torch.reshape(sobel,[1,1,3,3])
    filter = filter.cuda()
    gx = F.conv2d(img, filter, stride=1, padding=1)
    return gx

def gradient_y(img):
    sobel = torch.Tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    filter = torch.reshape(sobel, [1, 1,3,3])
    filter = filter.cuda()
    gy = F.conv2d(img, filter, stride=1, padding=1)
    return gy

def charbonnier_penalty(s):
    cp_s = torch.pow(torch.pow(s, 2) + 0.001**2, 0.5)
    return cp_s

def get_saliency_smoothness(pred, gt, size_average=True):
    alpha = 10
    s1 = 10
    s2 = 0
    ## first oder derivative: sobel
    sal_x = torch.abs(gradient_x(pred))
    sal_y = torch.abs(gradient_y(pred))
    gt_x = gradient_x(gt)
    gt_y = gradient_y(gt)
    w_x = torch.exp(torch.abs(gt_x) * (-alpha))
    w_y = torch.exp(torch.abs(gt_y) * (-alpha))
    cps_x = charbonnier_penalty(sal_x * w_x)
    cps_y = charbonnier_penalty(sal_y * w_y)
    cps_xy = cps_x + cps_y

    ## second order derivative: laplacian
    lap_sal = torch.abs(laplacian_edge(pred))
    lap_gt = torch.abs(laplacian_edge(gt))
    weight_lap = torch.exp(lap_gt * (-alpha))
    weighted_lap = charbonnier_penalty(lap_sal*weight_lap)

    smooth_loss = s1*torch.mean(cps_xy) + s2*torch.mean(weighted_lap)

    return smooth_loss

class smoothness_loss(torch.nn.Module):
    def __init__(self, size_average = True):
        super(smoothness_loss, self).__init__()
        self.size_average = size_average

    def forward(self, pred, target):

        return get_saliency_smoothness(pred, target, self.size_average)

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).sum()

class flooding_dice_bce_loss(torch.nn.Module):
    def __init__(self, batch=True, flooding_level=0.01):
        super(flooding_dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = torch.nn.BCELoss()
        self.flooding_level = flooding_level

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        return (a - self.flooding_level).abs() + self.flooding_level