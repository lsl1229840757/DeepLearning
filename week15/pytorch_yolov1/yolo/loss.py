import torch
import torch.nn.functional as F
import torch.nn as nn
from yolo.encoder import yolo_encoder


class yolov1_loss(nn.Module):
    def __init__(self, l_coord, l_obj, l_noobj):
        super(yolov1_loss, self).__init__()
        self.l_coord = l_coord
        self.l_noobj = l_noobj
        self.l_obj = l_obj

    def _prepare_target(self,meta,ceil_size,bbox_num,cls_num,device):
        target_cls = []
        target_obj = []
        target_box = []
        for target in meta:
            t = target['boxlist']
            t.resize(ceil_size)  # 这里把bbox的坐标全部转换到7*7的ceil中
            cls,obj,box = yolo_encoder(t,ceil_size,bbox_num,cls_num)  # encoder的作用就是把真实cls, response, bbox标记记录下来
            target_cls.append(torch.from_numpy(cls).unsqueeze(dim=0).float()) # cls shape: [20, 7, 7], 记录在哪个ceil出现了哪个类
            target_obj.append(torch.from_numpy(obj).unsqueeze(dim=0).float())  # obj shape: [2, 7, 7], 记录了一个ceil中的两个bbox是否reponse某个obj
            target_box.append(torch.from_numpy(box).unsqueeze(dim=0).float())  # bbox shape: [8, 7, 7], 记录了gt的中点和wh, 有两个bbox, 所以是8, 但是这两个bbox存一样的东西
        target_cls = torch.cat(target_cls).to(device)  # 前面多加1维是为了这里好cat成一个tensor
        target_obj = torch.cat(target_obj).to(device)
        target_box = torch.cat(target_box).to(device)
        return target_cls,target_obj,target_box  # 比前面多了一个batch维

    def offset2box(self,box):  # 这里是把(x,y,w,h)->(x1,y1,x2,y2): 前面那个是中心点的x,y

        box[:, 0] = box[:, 0] - box[:, 2] / 2
        box[:, 1] = box[:, 1] - box[:, 3] / 2
        box[:, 2] = box[:, 0] + box[:, 2]
        box[:, 3] = box[:, 1] + box[:, 3]

        return box

    def get_kp_torch_batch(self,pred, conf, topk=100):
        b, c, h, w = pred.shape
        pred = pred.contiguous().view(-1)
        pred[pred < conf] = 0
        score, topk_idx = torch.topk(pred, k=topk)

        # batch = topk_idx / (h * w * c)
        batch = torch.floor_divide(topk_idx, h*w*c)

        # cls = (topk_idx - batch * h * w * c) / (h * w)
        cls = torch.floor_divide(topk_idx - batch * h * w * c, h * w)

        channel = (topk_idx - batch * h * w * c) - (cls * h * w)

        x = channel % w
        # y = channel / w
        y = torch.floor_divide(channel, w)

        return x.view(-1), y.view(-1), cls.view(-1), batch.view(-1)

    def compute_iou(self,box1, box2):
        '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2]. [xmin, ymin, xmax, ymax]
        Args:
          box1: (tensor) bounding boxes, sized [N,4].
          box2: (tensor) bounding boxes, sized [M,4].
        Return:
          (tensor) iou, sized [N,M].
        '''

        lt = torch.max(
            box1[:, :2],  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, :2],  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = torch.min(
            box1[:, 2:],  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:, 2:],  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        wh[wh < 0] = 0  # clip at 0
        inter = wh[:, 0] * wh[:, 1]  # [N,M]

        area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
        area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

        iou = inter / (area1 + area2 - inter + 1e-4)
        return iou

    def forward(self,pred,meta):

        pred_cls, pred_response, pred_bboxes = pred
        device = pred_cls.get_device()

        B_size,cls_num,h,w = pred_cls.shape
        bbox_num = pred_response.shape[1]


        ceil_size = (w,h)
        label_cls, label_response, label_bboxes = self._prepare_target(meta,ceil_size,bbox_num,cls_num,device)

        device = pred_cls.get_device()
        label_cls = label_cls.to(device)
        label_response = label_response.to(device)
        label_bboxes = label_bboxes.to(device)

        with torch.no_grad():
            tmp_response = label_response.sum(dim=1).unsqueeze(dim=1)  # 把每个ceil的bbox的response加起来
            k = (tmp_response>0.9).sum()
            x_list,y_list,c_list,b_list = self.get_kp_torch_batch(tmp_response,conf=0.5,topk=int(k))

        t_responses = label_response[b_list, :, y_list, x_list]  # 把有物体的ceil中信息取出来
        p_responses = pred_response[b_list, :, y_list, x_list]

        t_boxes = label_bboxes[b_list, :, y_list, x_list]
        p_boxes = pred_bboxes[b_list, :, y_list, x_list]

        t_classes = label_cls[b_list, :, y_list, x_list]
        p_classes = pred_cls[b_list, :, y_list, x_list]

        loss_pos_cls = F.mse_loss(p_classes,t_classes, reduction='sum')  # 这是原文中的第3项


        t_offset = t_boxes.view(-1, 4)
        p_offset = p_boxes.view(-1, 4)
        with torch.no_grad():
            t_box = self.offset2box(t_offset.clone().float()).to(device)
            p_box = self.offset2box(p_offset.clone().float()).to(device)
            iou = self.compute_iou(t_box, p_box).view(-1,bbox_num)

        idx = iou.argmax(dim=1)
        idx = idx.unsqueeze(dim=1)
        gp_responses = p_responses.gather(1,idx)
        g_iou = iou.gather(1,idx)  # 因为这个部分是Pr(obj)*IOU 所以需要这么写
        loss_pos_response = F.mse_loss(gp_responses, g_iou, reduction='sum')  # 这是原文中的第5项

        idx = idx.unsqueeze(dim=1)
        p_boxes = p_boxes.view(-1, bbox_num, 4)
        t_boxes = t_boxes.view(-1, bbox_num, 4)
        off_idx = idx.repeat(1,1,4)  # [6, 1, 4], 6个有物体的ceil
        gp_boxes = p_boxes.gather(1,off_idx)  # [6, 2, 4] - > [6, 1, 4], 从6个ceil的2个box中抽iou最大的那个box
        gt_boxes = t_boxes.gather(1,off_idx)
        loss_pos_offset = F.mse_loss(gp_boxes, gt_boxes, reduction='sum')  # 这一项是原文中的第1, 2项, 只不过原文中的w,h开了根号



        neg_mask = label_response < 1
        neg_pred = pred_response[neg_mask]
        neg_target = label_response[neg_mask]

        loss_neg_response = F.mse_loss(neg_pred, neg_target, reduction='sum') / B_size * self.l_noobj  # 这是原文的第四项
        loss_pos_response = loss_pos_response / B_size * self.l_obj
        loss_pos_offset = loss_pos_offset / B_size * self.l_coord
        loss_pos_cls = loss_pos_cls / B_size


        return {'pObj': loss_pos_response,
                'nObj':loss_neg_response,
                'cls': loss_pos_cls,
                'offset': loss_pos_offset}
