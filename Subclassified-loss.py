import torch
import torch.nn as nn
import torch.nn.functional as F 


class CrossEntropyLoss_Sub(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce'):
        super(CrossEntropyLoss_Sub, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name


    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None

        if len(cls_score.size()) == 4:
            ####### BXCXHXW #######
            cluster_weight = 1.0
            ss = cls_score.size()
            down_sample_h = 8
            down_sample_w = 8
            ds = 2

            ########## The calculation of the weight map does not require gradients. ###########
            with torch.no_grad():
                ########## To better utilize the GPU memory, we downsample the image into two different resolutions for operation. #############
                #########################  Downsampling for computational efficiency  #############################
                cls_score_down = F.interpolate(cls_score, (ss[2]//down_sample_h, ss[3]//down_sample_w), mode="bilinear")
                cls_score_sup = F.interpolate(cls_score, (ss[2]//(down_sample_h//ds), ss[3]//(down_sample_w//ds)), mode="bilinear")

                t_label_down = F.interpolate(label.unsqueeze(1).float(), (ss[2]//down_sample_h, ss[3]//down_sample_w), mode="nearest").view(ss[0], -1).long()
                t_label_sup = F.interpolate(label.unsqueeze(1).float(), (ss[2]//(down_sample_h//ds), ss[3]//(down_sample_w//ds)), mode="nearest")

                pixel_weights = []
                #########  Block matrix multiplication to reduce memory consumption   #############
                splits = 1
                cls_score_down = cls_score_down.view(ss[0], ss[1], -1)
                ss2 = cls_score_sup.size()
                trunk = ss2[2] // splits
                for ii in range(splits):
                    cls_score_T = cls_score_sup[:,:, ii*trunk:(ii+1)*trunk, :].view(ss[0], ss[1], -1)

                    mmatrix_down = torch.sum(cls_score_down*cls_score_down, dim=1, keepdim=True)
                    mmatrix_sup = torch.sum(cls_score_T*cls_score_T, dim=1, keepdim=True).permute(0, 2, 1)
                    sm_down = mmatrix_down.size()
                    sm_sup = mmatrix_sup.size()
                    mmatrix_down = mmatrix_down.repeat(1, sm_sup[1], 1)
                    mmatrix_sup = mmatrix_sup.repeat(1, 1, sm_down[1])
                    mmask = (mmatrix_down > mmatrix_sup).float()
                    mmatrix = mmatrix_down * mmask + mmatrix_sup * (1.0 - mmask)

                    cos_sim = torch.bmm(cls_score_T.permute(0,2,1), cls_score_down) / (mmatrix + 1e-5)
                    t_label = t_label_sup[:,:, ii*trunk:(ii+1)*trunk, :].view(ss[0], -1).long()
                    class_mask = (t_label.unsqueeze(2) == t_label_down.unsqueeze(1).repeat(1, ss[2]*ss[3] * ds*ds // down_sample_h // down_sample_w // splits, 1)).long().float()
                    pixel_weight =  1. / torch.pow(torch.sum(cos_sim * class_mask > 0.98, dim=2).float() + 1e1, 0.5)
                    pixel_weights.append(pixel_weight)

                pixel_weight = torch.cat(pixel_weights, dim=1)          
                pixel_weight = pixel_weight.view(ss[0], 1, ss[2]//(down_sample_h//ds), ss[3]//(down_sample_w//ds))
                pixel_weight = F.interpolate(pixel_weight, size=(ss[2], ss[3]), mode="bilinear")

            loss_cls = self.loss_weight * self.cls_criterion(
                cls_score,
                label,
                weight,
                class_weight=class_weight,
                reduction="none",
                avg_factor=avg_factor,
                **kwargs)
            
            ############ weight normlization ############
            pixel_weight = torch.clamp(pixel_weight / torch.mean(pixel_weight, dim=(2,3), keepdim=True), max=10.0)

            loss_cls = torch.mean(loss_cls * pixel_weight.detach())
        else :
            loss_cls = self.loss_weight * self.cls_criterion(
                cls_score,
                label,
                weight,
                class_weight=class_weight,
                reduction=reduction,
                avg_factor=avg_factor,
                **kwargs)

        return loss_cls
