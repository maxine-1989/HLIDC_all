import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param reduction: `none`|`mean`|`sum`
    """

    def __init__(self, alpha=1, gamma=2, reduction="mean", **kwargs):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6  # set '1e-4' when train with FP16
        self.reduction = reduction

        assert self.reduction in ["none", "mean", "sum"]

    def forward(self, output, target):
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        target = target.unsqueeze(dim=1)
        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)  # / (torch.sum(pos_weight) + 1e-4)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = (
            -self.alpha * neg_weight * F.logsigmoid(-output)
        )  # / (torch.sum(neg_weight) + 1e-4)

        loss = pos_loss + neg_loss
        loss = loss.mean()
        return loss


class MultiFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=None, gamma=2, reduction="mean"):
        super(MultiFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = (
                torch.ones(
                    num_class,
                )
                - 0.5
            )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError("the length not equal to number of class")

    def forward(self, logits, targets):

        # print("logits:", logits.shape, "targets:", targets.shape)

        '''
        alpha = self.alpha.to(logit.device)
        prob = F.softmax(logit)
        '''
        # if prob.dim() > 2:
        #     # used for 3d-conv:  N,C,d1,d2 -> N,C,m (m=d1*d2*...)
        #     N, C = logit.shape[:2]
        #     prob = prob.view(N, C, -1)
        #     prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
        #     prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]

        # ori_shp = target.shape
        # target = target.view(-1, 1)

        # prob = prob.gather(1, target).view(-1) + self.smooth  # avoid nan
        # logpt = torch.log(prob)
        # # alpha_class = alpha.gather(0, target.squeeze(-1))
        # alpha_weight = alpha[target.squeeze().long()]
        # loss = -alpha_weight * torch.pow(torch.sub(1.0, prob), self.gamma) * logpt

        # if self.reduction == "mean":
        #     loss = loss.mean()
        # elif self.reduction == "none":
        #     loss = loss.view(ori_shp)

        '''
        pos_mask = target  # Mask where target is 1
        neg_mask = 1 - target  # Mask where target is 0

        pos_weight = (pos_mask * torch.pow(1 - prob, self.gamma)).detach()
        pos_loss = -pos_weight * torch.log(prob)

        neg_weight = (neg_mask * torch.pow(prob, self.gamma)).detach()
        neg_loss = -self.alpha * neg_weight * F.logsigmoid(-output)

        loss = pos_loss + neg_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss
        '''
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)  # [batch_size, num_classes]

        # Gather the probabilities of the target classes
        batch_indices = torch.arange(logits.size(0),device=logits.device)
        probs_target = probs[batch_indices, targets]  # [batch_size]

        # Compute the log-probabilities for the target classes
        log_probs = torch.log(probs_target + 1e-12)  # Avoid log(0)

        # Compute alpha_t
        alpha_t = self.alpha.to(logits.device)[targets] # [batch_size]

        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = torch.pow(1 - probs_target, self.gamma)

        # Compute Focal Loss
        loss = -alpha_t * focal_weight * log_probs  # [batch_size]

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss  # No reduction


class MultiLabelFocalLoss(nn.Module):
    """
    Focal_Loss= -1*alpha*((1-pt)**gamma)*log(pt)
    Args:
        num_class: number of classes
        alpha: class balance factor shape=[num_class, ]
        gamma: hyper-parameter
        reduction: reduction type
    """

    def __init__(self, num_class, alpha=0.7, gamma=2, reduction="mean"):
        super(MultiLabelFocalLoss, self).__init__()
        self.num_class = num_class
        self.gamma = gamma
        self.reduction = reduction
        self.smooth = 1e-4
        self.alpha = alpha
        if alpha is None:
            self.alpha = (
                torch.ones(
                    num_class,
                )
                - 0.5
            )
        elif isinstance(alpha, (int, float)):
            self.alpha = torch.as_tensor([alpha] * num_class)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.as_tensor(alpha)
        if self.alpha.shape[0] != num_class:
            raise RuntimeError("the length not equal to number of class")

    def forward(self, logits, targets):

        # print("logits:", logits.shape, "targets:", targets.shape)

        alpha = self.alpha.to(logits.device)
        probs = torch.sigmoid(logits)


        pos_mask = targets
        pos_weight = (1 - probs) ** self.gamma
        pos_loss = - alpha * pos_weight * torch.log(probs + self.smooth)

        neg_mask = 1 -targets
        neg_weight = probs ** self.gamma
        neg_loss = - (1 - alpha) * neg_weight * torch.log(1 -probs + self.smooth)

        loss = pos_mask * pos_loss + neg_mask * neg_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else: 
            return loss
