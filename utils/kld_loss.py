import torch
import torch.nn as nn

class KLDloss(nn.Module):

    def __init__(self, taf=1.0, fun="sqrt"):
        super(KLDloss, self).__init__()
        self.fun = fun
        self.taf = taf

    def forward(self, pred, target): # pred [[x,y,w,h,angle], ...]
        #assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 5)
        target = target.view(-1, 5)

        delta_x = pred[:, 0] - target[:, 0]
        delta_y = pred[:, 1] - target[:, 1]
        pre_angle_radian = pred[:, 4]
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

        

        if self.fun == "sqrt":
            kld = kld.clamp(1e-7).sqrt()
        elif self.fun == "log1p":
            kld = torch.log1p(kld.clamp(1e-7))
        else:
            pass

        kld_loss = 1 - 1 / (self.taf + kld)

        return kld_loss


class KLDloss_new(nn.Module):
    def __init__(self, taf = 2.0, alpha = 1.0, fun = 'sqrt', compute= 'KLD',reduction = "none"):
        super(KLDloss_new, self).__init__()
        self.taf = taf
        self.alpha = alpha
        self.fun = fun
        self.compute = compute
        self.reduction = reduction

    def KLD_compute(self, pred, target):  # pred [[x,y,w,h,angle], ...] [-pi/2,pi/2]
        xy_p, Sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        xy_t, Sigma_t = self.xy_wh_r_2_xy_sigma(target)
        _shape = xy_p.shape
  
        xy_p = xy_p.reshape(-1, 2)
        xy_t = xy_t.reshape(-1, 2)
        Sigma_p = Sigma_p.reshape(-1, 2, 2)
        Sigma_t = Sigma_t.reshape(-1, 2, 2)

        #assert (torch.linalg.matrix_rank(Sigma_p) == 2).min()>0, "rank(sigma_p) must full"
        Sigma_p_inv = torch.stack((Sigma_p[..., 1, 1], -Sigma_p[..., 0, 1],
                                -Sigma_p[..., 1, 0], Sigma_p[..., 0, 0]),
                                dim=-1).reshape(-1, 2, 2)
        Sigma_p_inv = Sigma_p_inv / Sigma_p.det().unsqueeze(-1).unsqueeze(-1)
        #Sigma_p_inv = torch.inverse(Sigma_p)

        dxy = (xy_p - xy_t).unsqueeze(-1)
        xy_distance = 0.5 * dxy.permute(0, 2, 1).bmm(Sigma_p_inv).bmm(dxy).view(-1)

        whr_distance = 0.5 * Sigma_p_inv.bmm(Sigma_t).diagonal(dim1=-2, dim2=-1).sum(dim=-1)

        Sigma_p_det_log = Sigma_p.det().clamp(1e-7).log()
        Sigma_t_det_log = Sigma_t.det().clamp(1e-7).log()
        #Sigma_p_det_log = torch.where(torch.isnan(Sigma_p_det_log), torch.full_like(Sigma_p_det_log, 10), Sigma_p_det_log) 
        #Sigma_t_det_log = torch.where(torch.isnan(Sigma_t_det_log), torch.full_like(Sigma_t_det_log, 10), Sigma_t_det_log)
        #distance = xy_distance / (alpha * alpha) + whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log) - 1
        whr_distance = whr_distance + 0.5 * (Sigma_p_det_log - Sigma_t_det_log)
        whr_distance = whr_distance - 1
        distance = (xy_distance / (self.alpha * self.alpha) + whr_distance)
        #distance = torch.where(torch.isnan(distance), torch.full_like(distance, 0), distance) 
        
        if self.fun == 'sqrt':
            distance = distance.clamp(1e-7).sqrt()
        elif self.fun == 'log1p':
            distance = torch.log1p(distance)
        else:
            pass  #distance = torch.log1p(distance.clamp(1e-7))
        
        distance = distance.reshape(_shape[:-1])

        return distance

    def BD_compute(self, pred, target):

        mu_p, sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        mu_t, sigma_t = self.xy_wh_r_2_xy_sigma(target)

        mu_p = mu_p.reshape(-1, 2)
        mu_t = mu_t.reshape(-1, 2)
        sigma_p = sigma_p.reshape(-1, 2, 2)
        sigma_t = sigma_t.reshape(-1, 2, 2)

        delta = (mu_p - mu_t).unsqueeze(-1)
        sigma = 0.5 * (sigma_p + sigma_t)
        #sigma_inv = torch.inverse(sigma)
        Sigma_inv = torch.stack((sigma[..., 1, 1], -sigma[..., 0, 1],
                                -sigma[..., 1, 0], sigma[..., 0, 0]),
                                dim=-1).reshape(-1, 2, 2)
        sigma_inv = Sigma_inv / sigma.det().unsqueeze(-1).unsqueeze(-1)

        term1 = torch.log(
            torch.det(sigma) /
            (torch.sqrt(torch.det(sigma_t.matmul(sigma_p))))).reshape(-1, 1)
        term2 = delta.transpose(-1, -2).matmul(sigma_inv).matmul(delta).squeeze(-1)
        dis = 0.5 * term1 + 0.125 * term2
        bcd_dis = dis.clamp(min=1e-7)
        
        if self.fun == 'sqrt':
            bcd_dis = bcd_dis.sqrt()
        elif self.fun == 'log1p':
            bcd_dis = torch.log1p(bcd_dis)
        else:
            pass

        return bcd_dis

    def GWD_compute(self, pred, target):

        mu_p, sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        mu_t, sigma_t = self.xy_wh_r_2_xy_sigma(target)

        xy_distance = (mu_p - mu_t).square().sum(dim=-1)

        whr_distance = sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        whr_distance = whr_distance + sigma_t.diagonal(
            dim1=-2, dim2=-1).sum(dim=-1)

        _t_tr = (sigma_p.bmm(sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        _t_det_sqrt = (sigma_p.det() * sigma_t.det()).clamp(0).sqrt()
        whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clamp(0).sqrt()

        dis = xy_distance + whr_distance
        gwd_dis = dis.clamp(min=1e-6)

        if self.fun == 'sqrt':
            gwd_dis = gwd_dis.sqrt()
        elif self.fun == 'log1p':
            gwd_dis = torch.log1p(gwd_dis)
        else:
            pass

        return gwd_dis

    def forward(self, pred, target):
        assert self.reduction in ['none', 'min', 'max', 'mean']
        assert self.compute in ['KLD','BD','GWD']
        if self.compute == 'KLD':
            kld_pt_loss = self.KLD_compute(pred, target)
            if self.reduction == 'none':
                kld = kld_pt_loss
            elif self.reduction == 'mean':
                kld_tp_loss = self.KLD_compute(target, pred)
                kld = 0.5 * (kld_pt_loss + kld_tp_loss)
            elif self.reduction == 'min':
                kld_tp_loss = self.KLD_compute(target, pred)
                kld = torch.min(kld_pt_loss, kld_tp_loss)
            else:  # 'max'
                kld_tp_loss = self.KLD_compute(target, pred)
                kld = torch.max(kld_pt_loss, kld_tp_loss)
            kld_loss = 1 - 1 / (self.taf + kld)#kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))
            return kld_loss , kld
        elif self.compute == 'BD':
            bd_pt_loss = self.BD_compute(pred, target)
            if self.reduction == 'none':
                bd = bd_pt_loss
            elif self.reduction == 'mean':
                bd_tp_loss = self.BD_compute(target, pred)
                bd = 0.5 * (bd_pt_loss + bd_tp_loss)
            elif self.reduction == 'min':
                bd_tp_loss = self.BD_compute(target, pred)
                bd = torch.min(bd_pt_loss, bd_tp_loss)
            else:  # 'max'
                bd_tp_loss = self.BD_compute(target, pred)
                bd = torch.max(bd_pt_loss, bd_tp_loss)
            bd_loss = 1 - 1 / (self.taf + bd)#kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))
            return bd_loss , bd
        else:
            gwd_pt_loss = self.GWD_compute(pred, target)
            if self.reduction == 'none':
                gwd = gwd_pt_loss
            elif self.reduction == 'mean':
                gwd_tp_loss = self.GWD_compute(target, pred)
                gwd = 0.5 * (gwd_pt_loss + gwd_tp_loss)
            elif self.reduction == 'min':
                gwd_tp_loss = self.GWD_compute(target, pred)
                gwd = torch.min(gwd_pt_loss, gwd_tp_loss)
            else:  # 'max'
                gwd_tp_loss = self.GWD_compute(target, pred)
                gwd = torch.max(gwd_pt_loss, gwd_tp_loss)

            gwd_loss = 1 - 1 / (self.taf + gwd) #kld_loss = 1 - 1 / (self.taf + torch.log(kld + 1))
            return gwd_loss , gwd # 1/2+gwd
        

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
        # # add raodong
        # wh[:,0] += torch.rand(1)
        # wh[:,1] += torch.rand(1)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy.type(xywhr.type()), sigma.type(xywhr.type())

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
        return xy.type(xy_stddev_pearson.type()), sigma.type(xy_stddev_pearson.type())
    
def compute_kld_loss(targets, preds,taf=1.0,fun='sqrt'):
    with torch.no_grad():
        kld_loss_ts_ps = torch.zeros(0, preds.shape[0], device=targets.device)
        for target in targets:
            target = target.unsqueeze(0).repeat(preds.shape[0], 1)
            kld_loss_t_p = kld_loss(preds, target,taf=taf, fun=fun)
            kld_loss_ts_ps = torch.cat((kld_loss_ts_ps, kld_loss_t_p.unsqueeze(0)), dim=0)
    return kld_loss_ts_ps


def kld_loss(pred, target, taf=1.0, fun='sqrt'):  # pred [[x,y,w,h,angle], ...]
    #assert pred.shape[0] == target.shape[0]

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

    if fun == "sqrt":
        kld = kld.clamp(1e-7).sqrt()
    elif fun == "log1p":
        kld = torch.log1p(kld.clamp(1e-7))
    else:
        pass

    kld_loss = 1 - 1 / (taf + kld)
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
