import torch
import torch.nn as nn

class KFiou(nn.Module):

    def __init__(self, fun='none',
                 beta=1.0 / 9.0,
                 eps=1e-6):
        super(KFiou, self).__init__()
        self.eps = eps
        self.beta = beta
        self.fun = fun

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        pred = pred.type(torch.float32)
        target = target.type(torch.float32)
        xy_p, Sigma_p = self.xy_wh_r_2_xy_sigma(pred)
        xy_t, Sigma_t = self.xy_wh_r_2_xy_sigma(target)

        # Smooth-L1 norm
        diff = torch.abs(xy_p - xy_t)
        xy_loss = torch.where(diff < self.beta, 0.5 * diff * diff / self.beta,
                              diff - 0.5 * self.beta).sum(dim=-1)
        Vb_p = 4 * Sigma_p.det().clamp(1e-7).sqrt()
        Vb_t = 4 * Sigma_t.det().clamp(1e-7).sqrt()
        K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
        Sigma = Sigma_p - K.bmm(Sigma_p)
        Vb = 4 * Sigma.det().clamp(1e-7).sqrt()
        Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
        #Vb_p = torch.where(torch.isnan(Vb_p), torch.full_like(Vb_p, 0), Vb_p)
        #Vb_t = torch.where(torch.isnan(Vb_p), torch.full_like(Vb_t, 0), Vb_t)
        KFIoU = Vb / (Vb_p + Vb_t - Vb + self.eps) #Vb_p 有nan 

        if self.fun == 'ln':
            kf_loss = - torch.log(KFIoU + self.eps)
        elif self.fun == 'exp':
            kf_loss = torch.exp(1 - KFIoU) - 1
        else:
            kf_loss = 1 - KFIoU

        loss = (0.01 * xy_loss + kf_loss).clamp(1e-7)  #xy_loss 因为用的解码之后的 xy_loss 刚开始回比较大 所以直接缩小10倍
        KFIoU =  1 / (1 + torch.log1p(loss)) # use yolov iou 作为obj的rotion
        return loss, KFIoU

    def xy_wh_r_2_xy_sigma(self, xywhr):
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
        r =  xywhr[..., 4]
        # 弧度制
        # r = (3.141592 * xywhr[..., 4]) / 180.0
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy.type(torch.float32), sigma.type(torch.float32)
