import torch
import torch.nn as nn

class KLDloss(nn.Module):

    def __init__(self, taf=1.0, reduction="none"):
        super(KLDloss, self).__init__()
        self.reduction = reduction
        self.taf = taf

    def forward(self, pred, target): # pred [[x,y,w,h,angle], ...]
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 5)
        target = target.view(-1, 5)

        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        #角度制-弧度制
        #pre_angle_radian = 3.141592653589793 * pred[:, 4] / 180.0
        pre_angle_radian = pred[:, 4]
        #targrt_angle_radian = 3.141592653589793 * target[:, 4] / 180.0
        targrt_angle_radian = target[:, 4]
        delta_angle_radian = pre_angle_radian - targrt_angle_radian

        kld =  0.5 * (
                        4 * torch.pow( ( delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 2], 2)
                      + 4 * torch.pow( ( delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian)) ), 2) / torch.pow(target[:, 3], 2)
                     )\
             + 0.5 * (
                        torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                      + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                      + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                     )\
             + 0.5 * (
                        torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                      + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
                     )\
             - 1.0

        kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))

        # if self.reduction == "mean":
        #     kld_loss = loss.mean()
        # elif self.reduction == "sum":
        #     kld_loss = loss.sum()

        return kld_loss


class KLDloss_new(nn.Module):# 对类正方形的loss 计算有误 角度信息直接丢失了
    def __init__(self, taf=1.0,alpha=1.0, fun = 'sqrt',reduction="none"):
        super(KLDloss_new, self).__init__()
        self.taf = taf
        self.alpha = alpha
        self.fun = fun
        self.reduction = reduction

    def KLD_compute(self, pred, target):  # pred [[x,y,w,h,angle], ...] 角度都在[-pi/2,pi/2]
        assert pred.shape[0] == target.shape[0]

        # mu_p, sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        # mu_t, sigma_t = self.xy_wh_r_2_xy_sigma(target)

        # mu_p = mu_p.reshape(-1, 2)
        # mu_t = mu_t.reshape(-1, 2)
        # sigma_p = sigma_p.reshape(-1, 2, 2)
        # sigma_t = sigma_t.reshape(-1, 2, 2)

        # delta = (mu_p - mu_t).unsqueeze(-1)
        # sigma_t_inv = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
        #                            -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
        #                           dim=-1).reshape(-1, 2, 2) # torch.linalg.inv(sigma_t)
        # term1 = delta.transpose(-1,
        #                         -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)
        # term2 = torch.diagonal(
        #     sigma_t_inv.matmul(sigma_p),
        #     dim1=-2, dim2=-1).sum(dim=-1, keepdim=True) + \
        #     torch.log(torch.det(sigma_t) / torch.det(sigma_p)).reshape(-1, 1)
        # #term1 = torch.where(torch.isnan(term1), torch.full_like(term1, 0), term1)
        # #term2 = torch.where(torch.isnan(term2), torch.full_like(term2, 0), term2) #人为修正 
        # dis = term1 + term2 - 2
        # kl_dis = dis.clamp(min=1e-6)
        # if sqrt:
        #     kl_dis =  torch.sqrt(kl_dis)
        # else:
        #     kl_dis =  torch.log1p(kl_dis)
        # return kl_dis.squeeze()

        # ##old##
        xy_p, Sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        xy_t, Sigma_t = self.xy_wh_r_2_xy_sigma(target)
        _shape = xy_p.shape
        # 这两句都没必要
        Sigma_p = Sigma_p.reshape(-1, 2, 2)
        Sigma_t = Sigma_t.reshape(-1, 2, 2)

        Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                                   -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                                  dim=-1).reshape(-1, 2, 2)

        Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)

        dxy = (xy_p - xy_t).unsqueeze(-1)
        xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

        whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        Sigma_p_det_log = Sigma_p.det().log()
        Sigma_t_det_log = Sigma_t.det().log()
        #Sigma_p_det_log = torch.where(torch.isnan(Sigma_p_det_log), torch.full_like(Sigma_p_det_log, 10), Sigma_p_det_log) #人为修正
        #Sigma_t_det_log = torch.where(torch.isnan(Sigma_t_det_log), torch.full_like(Sigma_t_det_log, 10), Sigma_t_det_log)
        #distance = xy_distance / (alpha * alpha) + whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log) - 1
        whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
        whr_distance = whr_distance - 1
        distance = (xy_distance / (self.alpha * self.alpha) + whr_distance)
        #distance = torch.where(torch.isnan(distance), torch.full_like(distance, 0), distance) #人为修正
        
        if self.fun == 'sqrt':
            distance = distance.clamp(1e-7).sqrt()
        elif self.fun == 'log1p':
            distance = torch.log1p(distance.clamp(1e-7))
        else:
            pass  #distance = torch.log1p(distance.clamp(1e-7))
        distance = distance.reshape(_shape[:-1])

        return distance

    def forward(self, pred, target):
        assert self.reduction in ['none', 'min', 'max', 'mean']
        kld_pt_loss = self.KLD_compute(pred, target)
        if self.reduction == 'none':
            kld = kld_pt_loss
        if self.reduction == 'mean':
            kld_tp_loss = self.KLD_compute(target, pred)
            kld = 0.5 * (kld_pt_loss + kld_tp_loss)
        elif self.reduction == 'min':
            kld_tp_loss = self.KLD_compute(target, pred)
            kld = torch.min(kld_pt_loss, kld_tp_loss)
        else:  # 'max'
            kld_tp_loss = self.KLD_compute(target, pred)
            kld = torch.max(kld_pt_loss, kld_tp_loss)

        kld_loss = 1 - 1 / (self.taf + kld)#kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))

        return kld_loss #, 1 / (self.taf + kld)

    def xy_wh_r_2_xy_sigma(self,xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        # r = (3.141592 * xywhr[..., 4]) / 180.0 #角度转弧度
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma

    def xy_stddev_pearson_2_xy_sigma(self, xy_stddev_pearson):
        """Convert oriented bounding box from the Pearson coordinate system to 2-D
        Gaussian distribution.

        Args:
            xy_stddev_pearson (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xy_stddev_pearson.shape
        assert _shape[-1] == 5
        xy = xy_stddev_pearson[..., :2]
        stddev = xy_stddev_pearson[..., 2:4]
        pearson = xy_stddev_pearson[..., 4].clamp(min=1e-7 - 1, max=1 - 1e-7)
        covar = pearson * stddev.prod(dim=-1)
        var = stddev.square()
        sigma = torch.stack((var[..., 0], covar, covar, var[..., 1]),
                            dim=-1).reshape(_shape[:-1] + (2, 2))
        return xy, sigma

def compute_kld_loss(targets, preds):
    with torch.no_grad():
        kld_loss_ts_ps = torch.zeros(0, preds.shape[0], device=targets.device)
        for target in targets:
            target = target.unsqueeze(0).repeat(preds.shape[0], 1)
            kld_loss_t_p = kld_loss(preds, target)
            kld_loss_ts_ps = torch.cat((kld_loss_ts_ps, kld_loss_t_p.unsqueeze(0)), dim=0)
    return kld_loss_ts_ps


def kld_loss(pred, target, taf=1.0):  # pred [[x,y,w,h,angle], ...]
    assert pred.shape[0] == target.shape[0]

    pred = pred.view(-1, 5)
    target = target.view(-1, 5)

    delta_x = pred[:, 0] - target[:, 0]
    delta_y = pred[:, 1] - target[:, 1]
    pre_angle_radian = pred[:, 4]  #3.141592653589793 * pred[:, 4] / 180.0
    targrt_angle_radian = target[:, 4] #3.141592653589793 * target[:, 4] / 180.0
    delta_angle_radian = pre_angle_radian - targrt_angle_radian

    kld = 0.5 * (
            4 * torch.pow((delta_x.mul(torch.cos(targrt_angle_radian)) + delta_y.mul(torch.sin(targrt_angle_radian))),
                          2) / torch.pow(target[:, 2], 2)
            + 4 * torch.pow((delta_y.mul(torch.cos(targrt_angle_radian)) - delta_x.mul(torch.sin(targrt_angle_radian))),
                            2) / torch.pow(target[:, 3], 2)
    ) \
          + 0.5 * (
                  torch.pow(pred[:, 3], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.sin(delta_angle_radian), 2)
                  + torch.pow(pred[:, 3], 2) / torch.pow(target[:, 3], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
                  + torch.pow(pred[:, 2], 2) / torch.pow(target[:, 2], 2) * torch.pow(torch.cos(delta_angle_radian), 2)
          ) \
          + 0.5 * (
                  torch.log(torch.pow(target[:, 3], 2) / torch.pow(pred[:, 3], 2))
                  + torch.log(torch.pow(target[:, 2], 2) / torch.pow(pred[:, 2], 2))
          ) \
          - 1.0

    kld_loss = 1 - 1 / (taf + torch.log(kld + 1))

    return kld_loss

if __name__ == '__main__':
    '''
        测试损失函数
    '''
    kld_loss_n =KLDloss_new(alpha=1,fun='log1p')
    pred = torch.tensor([[5, 5, 5, 23, 0.15],[6,6,5,28,0]]).type(torch.float32)
    target = torch.tensor([[5, 5, 5, 24, 0],[6,6,5,28,0]]).type(torch.float32)
    kld = kld_loss_n(target,pred)
    #print(compute_kld_loss(target,pred))
    print(kld)
    #print(D)
    # D = torch.tensor(0)
    # print(torch.sigmoid(D))
    #print(torch.cos(pred[:,-1]))