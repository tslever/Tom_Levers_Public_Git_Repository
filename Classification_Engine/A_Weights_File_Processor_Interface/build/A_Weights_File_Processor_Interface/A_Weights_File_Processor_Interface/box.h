#ifndef BOX_H
#define BOX_H


#include "darknet.h"


box float_to_box(float* f);
box float_to_box_stride(float* f, int stride);
float box_iou(box a, box b);
float box_iou_kind(box a, box b, IOU_LOSS iou_kind);
float box_giou(box a, box b);
float box_diou(box a, box b);
float box_ciou(box a, box b);
dxrep dx_box_iou(box a, box b, IOU_LOSS iou_loss);
boxabs to_tblr(box a);
float box_rmse(box a, box b);


#endif