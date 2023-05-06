#include "pch.h"
#include "box.h"
#include <corecrt_math.h>
#include <corecrt_math_defines.h>


box float_to_box(float* f)
{
    box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    return b;
}


box float_to_box_stride(float* f, int stride)
{
    box b = { 0 };
    b.x = f[0];
    b.y = f[1 * stride];
    b.w = f[2 * stride];
    b.h = f[3 * stride];
    return b;
}


float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}


float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w * h;
    return area;
}


float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}


float box_iou(box a, box b)
{
    //return box_intersection(a, b)/box_union(a, b);

    float I = box_intersection(a, b);
    float U = box_union(a, b);
    if (I == 0 || U == 0) {
        return 0;
    }
    return I / U;
}


float box_iou_kind(box a, box b, IOU_LOSS iou_kind)
{
    //IOU, GIOU, MSE, DIOU, CIOU
    switch (iou_kind) {
    case IOU: return box_iou(a, b);
    case GIOU: return box_giou(a, b);
    case DIOU: return box_diou(a, b);
    case CIOU: return box_ciou(a, b);
    }
    return box_iou(a, b);
}


// where c is the smallest box that fully encompases a and b
boxabs box_c(box a, box b) {
    boxabs ba = { 0 };
    ba.top = fmin(a.y - a.h / 2, b.y - b.h / 2);
    ba.bot = fmax(a.y + a.h / 2, b.y + b.h / 2);
    ba.left = fmin(a.x - a.w / 2, b.x - b.w / 2);
    ba.right = fmax(a.x + a.w / 2, b.x + b.w / 2);
    return ba;
}


float box_giou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float u = box_union(a, b);
    float giou_term = (c - u) / c;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, giou_term: %f\n", c, u, giou_term);
#endif
    return iou - giou_term;
}


dxrep dx_box_iou(box pred, box truth, IOU_LOSS iou_loss) {
    boxabs pred_tblr = to_tblr(pred);
    float pred_t = fmin(pred_tblr.top, pred_tblr.bot);
    float pred_b = fmax(pred_tblr.top, pred_tblr.bot);
    float pred_l = fmin(pred_tblr.left, pred_tblr.right);
    float pred_r = fmax(pred_tblr.left, pred_tblr.right);
    //dbox dover = derivative(pred,truth);
    //dbox diouu = diou(pred, truth);
    boxabs truth_tblr = to_tblr(truth);
#ifdef DEBUG_PRINTS
    printf("\niou: %f, giou: %f\n", box_iou(pred, truth), box_giou(pred, truth));
    printf("pred: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", pred.x, pred.y, pred.w, pred.h, pred_tblr.top, pred_tblr.bot, pred_tblr.left, pred_tblr.right);
    printf("truth: x,y,w,h: (%f, %f, %f, %f) -> t,b,l,r: (%f, %f, %f, %f)\n", truth.x, truth.y, truth.w, truth.h, truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
#endif
    //printf("pred (t,b,l,r): (%f, %f, %f, %f)\n", pred_t, pred_b, pred_l, pred_r);
    //printf("trut (t,b,l,r): (%f, %f, %f, %f)\n", truth_tblr.top, truth_tblr.bot, truth_tblr.left, truth_tblr.right);
    dxrep ddx = { 0 };
    float X = (pred_b - pred_t) * (pred_r - pred_l);
    float Xhat = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
    float Ih = fmin(pred_b, truth_tblr.bot) - fmax(pred_t, truth_tblr.top);
    float Iw = fmin(pred_r, truth_tblr.right) - fmax(pred_l, truth_tblr.left);
    float I = Iw * Ih;
    float U = X + Xhat - I;
    float S = (pred.x - truth.x) * (pred.x - truth.x) + (pred.y - truth.y) * (pred.y - truth.y);
    float giou_Cw = fmax(pred_r, truth_tblr.right) - fmin(pred_l, truth_tblr.left);
    float giou_Ch = fmax(pred_b, truth_tblr.bot) - fmin(pred_t, truth_tblr.top);
    float giou_C = giou_Cw * giou_Ch;
    //float IoU = I / U;
//#ifdef DEBUG_PRINTS
    //printf("X: %f", X);
    //printf(", Xhat: %f", Xhat);
    //printf(", Ih: %f", Ih);
    //printf(", Iw: %f", Iw);
    //printf(", I: %f", I);
    //printf(", U: %f", U);
    //printf(", IoU: %f\n", I / U);
//#endif

    //Partial Derivatives, derivatives
    float dX_wrt_t = -1 * (pred_r - pred_l);
    float dX_wrt_b = pred_r - pred_l;
    float dX_wrt_l = -1 * (pred_b - pred_t);
    float dX_wrt_r = pred_b - pred_t;
    // UNUSED
    //// Ground truth
    //float dXhat_wrt_t = -1 * (truth_tblr.right - truth_tblr.left);
    //float dXhat_wrt_b = truth_tblr.right - truth_tblr.left;
    //float dXhat_wrt_l = -1 * (truth_tblr.bot - truth_tblr.top);
    //float dXhat_wrt_r = truth_tblr.bot - truth_tblr.top;

    // gradient of I min/max in IoU calc (prediction)
    float dI_wrt_t = pred_t > truth_tblr.top ? (-1 * Iw) : 0;
    float dI_wrt_b = pred_b < truth_tblr.bot ? Iw : 0;
    float dI_wrt_l = pred_l > truth_tblr.left ? (-1 * Ih) : 0;
    float dI_wrt_r = pred_r < truth_tblr.right ? Ih : 0;
    // derivative of U with regard to x
    float dU_wrt_t = dX_wrt_t - dI_wrt_t;
    float dU_wrt_b = dX_wrt_b - dI_wrt_b;
    float dU_wrt_l = dX_wrt_l - dI_wrt_l;
    float dU_wrt_r = dX_wrt_r - dI_wrt_r;
    // gradient of C min/max in IoU calc (prediction)
    float dC_wrt_t = pred_t < truth_tblr.top ? (-1 * giou_Cw) : 0;
    float dC_wrt_b = pred_b > truth_tblr.bot ? giou_Cw : 0;
    float dC_wrt_l = pred_l < truth_tblr.left ? (-1 * giou_Ch) : 0;
    float dC_wrt_r = pred_r > truth_tblr.right ? giou_Ch : 0;

    float p_dt = 0;
    float p_db = 0;
    float p_dl = 0;
    float p_dr = 0;
    if (U > 0) {
        p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
        p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
        p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
        p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
    }
    // apply grad from prediction min/max for correct corner selection
    p_dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
    p_db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
    p_dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
    p_dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

    if (iou_loss == GIOU) {
        if (giou_C > 0) {
            // apply "C" term from gIOU
            p_dt += ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
            p_db += ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
            p_dl += ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
            p_dr += ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
        }
        if (Iw <= 0 || Ih <= 0) {
            p_dt = ((giou_C * dU_wrt_t) - (U * dC_wrt_t)) / (giou_C * giou_C);
            p_db = ((giou_C * dU_wrt_b) - (U * dC_wrt_b)) / (giou_C * giou_C);
            p_dl = ((giou_C * dU_wrt_l) - (U * dC_wrt_l)) / (giou_C * giou_C);
            p_dr = ((giou_C * dU_wrt_r) - (U * dC_wrt_r)) / (giou_C * giou_C);
        }
    }

    float Ct = fmin(pred.y - pred.h / 2, truth.y - truth.h / 2);
    float Cb = fmax(pred.y + pred.h / 2, truth.y + truth.h / 2);
    float Cl = fmin(pred.x - pred.w / 2, truth.x - truth.w / 2);
    float Cr = fmax(pred.x + pred.w / 2, truth.x + truth.w / 2);
    float Cw = Cr - Cl;
    float Ch = Cb - Ct;
    float C = Cw * Cw + Ch * Ch;

    float dCt_dx = 0;
    float dCt_dy = pred_t < truth_tblr.top ? 1 : 0;
    float dCt_dw = 0;
    float dCt_dh = pred_t < truth_tblr.top ? -0.5 : 0;

    float dCb_dx = 0;
    float dCb_dy = pred_b > truth_tblr.bot ? 1 : 0;
    float dCb_dw = 0;
    float dCb_dh = pred_b > truth_tblr.bot ? 0.5 : 0;

    float dCl_dx = pred_l < truth_tblr.left ? 1 : 0;
    float dCl_dy = 0;
    float dCl_dw = pred_l < truth_tblr.left ? -0.5 : 0;
    float dCl_dh = 0;

    float dCr_dx = pred_r > truth_tblr.right ? 1 : 0;
    float dCr_dy = 0;
    float dCr_dw = pred_r > truth_tblr.right ? 0.5 : 0;
    float dCr_dh = 0;

    float dCw_dx = dCr_dx - dCl_dx;
    float dCw_dy = dCr_dy - dCl_dy;
    float dCw_dw = dCr_dw - dCl_dw;
    float dCw_dh = dCr_dh - dCl_dh;

    float dCh_dx = dCb_dx - dCt_dx;
    float dCh_dy = dCb_dy - dCt_dy;
    float dCh_dw = dCb_dw - dCt_dw;
    float dCh_dh = dCb_dh - dCt_dh;

    // UNUSED
    //// ground truth
    //float dI_wrt_xhat_t = pred_t < truth_tblr.top ? (-1 * Iw) : 0;
    //float dI_wrt_xhat_b = pred_b > truth_tblr.bot ? Iw : 0;
    //float dI_wrt_xhat_l = pred_l < truth_tblr.left ? (-1 * Ih) : 0;
    //float dI_wrt_xhat_r = pred_r > truth_tblr.right ? Ih : 0;

    // Final IOU loss (prediction) (negative of IOU gradient, we want the negative loss)
    float p_dx = 0;
    float p_dy = 0;
    float p_dw = 0;
    float p_dh = 0;

    p_dx = p_dl + p_dr;           //p_dx, p_dy, p_dw and p_dh are the gradient of IoU or GIoU.
    p_dy = p_dt + p_db;
    p_dw = (p_dr - p_dl);         //For dw and dh, we do not divided by 2.
    p_dh = (p_db - p_dt);

    // https://github.com/Zzh-tju/DIoU-darknet
    // https://arxiv.org/abs/1911.08287
    if (iou_loss == DIOU) {
        if (C > 0) {
            p_dx += (2 * (truth.x - pred.x) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) / (C * C);
            p_dy += (2 * (truth.y - pred.y) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) / (C * C);
            p_dw += (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C);
            p_dh += (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C);
        }
        if (Iw <= 0 || Ih <= 0) {
            p_dx = (2 * (truth.x - pred.x) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) / (C * C);
            p_dy = (2 * (truth.y - pred.y) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) / (C * C);
            p_dw = (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C);
            p_dh = (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C);
        }
    }
    //The following codes are calculating the gradient of ciou.

    if (iou_loss == CIOU) {
        float ar_gt = truth.w / truth.h;
        float ar_pred = pred.w / pred.h;
        float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
        float alpha = ar_loss / (1 - I / U + ar_loss + 0.000001);
        float ar_dw = 8 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * pred.h;
        float ar_dh = -8 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * pred.w;
        if (C > 0) {
            // dar*
            p_dx += (2 * (truth.x - pred.x) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) / (C * C);
            p_dy += (2 * (truth.y - pred.y) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) / (C * C);
            p_dw += (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C) + alpha * ar_dw;
            p_dh += (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C) + alpha * ar_dh;
        }
        if (Iw <= 0 || Ih <= 0) {
            p_dx = (2 * (truth.x - pred.x) * C - (2 * Cw * dCw_dx + 2 * Ch * dCh_dx) * S) / (C * C);
            p_dy = (2 * (truth.y - pred.y) * C - (2 * Cw * dCw_dy + 2 * Ch * dCh_dy) * S) / (C * C);
            p_dw = (2 * Cw * dCw_dw + 2 * Ch * dCh_dw) * S / (C * C) + alpha * ar_dw;
            p_dh = (2 * Cw * dCw_dh + 2 * Ch * dCh_dh) * S / (C * C) + alpha * ar_dh;
        }
    }

    ddx.dt = p_dx;      //We follow the original code released from GDarknet. So in yolo_layer.c, dt, db, dl, dr are already dx, dy, dw, dh.
    ddx.db = p_dy;
    ddx.dl = p_dw;
    ddx.dr = p_dh;

    // UNUSED
    //// ground truth
    //float gt_dt = ((U * dI_wrt_xhat_t) - (I * (dXhat_wrt_t - dI_wrt_xhat_t))) / (U * U);
    //float gt_db = ((U * dI_wrt_xhat_b) - (I * (dXhat_wrt_b - dI_wrt_xhat_b))) / (U * U);
    //float gt_dl = ((U * dI_wrt_xhat_l) - (I * (dXhat_wrt_l - dI_wrt_xhat_l))) / (U * U);
    //float gt_dr = ((U * dI_wrt_xhat_r) - (I * (dXhat_wrt_r - dI_wrt_xhat_r))) / (U * U);

    // no min/max grad applied
    //dx.dt = dt;
    //dx.db = db;
    //dx.dl = dl;
    //dx.dr = dr;

    //// sum in gt -- THIS DOESNT WORK
    //dx.dt += gt_dt;
    //dx.db += gt_db;
    //dx.dl += gt_dl;
    //dx.dr += gt_dr;

    //// instead, look at the change between pred and gt, and weight t/b/l/r appropriately...
    //// need the real derivative here (I think?)
    //float delta_t = fmax(truth_tblr.top, pred_t) - fmin(truth_tblr.top, pred_t);
    //float delta_b = fmax(truth_tblr.bot, pred_b) - fmin(truth_tblr.bot, pred_b);
    //float delta_l = fmax(truth_tblr.left, pred_l) - fmin(truth_tblr.left, pred_l);
    //float delta_r = fmax(truth_tblr.right, pred_r) - fmin(truth_tblr.right, pred_r);

    //dx.dt *= delta_t / (delta_t + delta_b);
    //dx.db *= delta_b / (delta_t + delta_b);
    //dx.dl *= delta_l / (delta_l + delta_r);
    //dx.dr *= delta_r / (delta_l + delta_r);

    // UNUSED
    //// ground truth
    //float gt_dt = ((U * dI_wrt_xhat_t) - (I * (dXhat_wrt_t - dI_wrt_xhat_t))) / (U * U);
    //float gt_db = ((U * dI_wrt_xhat_b) - (I * (dXhat_wrt_b - dI_wrt_xhat_b))) / (U * U);
    //float gt_dl = ((U * dI_wrt_xhat_l) - (I * (dXhat_wrt_l - dI_wrt_xhat_l))) / (U * U);
    //float gt_dr = ((U * dI_wrt_xhat_r) - (I * (dXhat_wrt_r - dI_wrt_xhat_r))) / (U * U);

    // no min/max grad applied
    //dx.dt = dt;
    //dx.db = db;
    //dx.dl = dl;
    //dx.dr = dr;

    // apply grad from prediction min/max for correct corner selection
    //dx.dt = pred_tblr.top < pred_tblr.bot ? p_dt : p_db;
    //dx.db = pred_tblr.top < pred_tblr.bot ? p_db : p_dt;
    //dx.dl = pred_tblr.left < pred_tblr.right ? p_dl : p_dr;
    //dx.dr = pred_tblr.left < pred_tblr.right ? p_dr : p_dl;

    //// sum in gt -- THIS DOESNT WORK
    //dx.dt += gt_dt;
    //dx.db += gt_db;
    //dx.dl += gt_dl;
    //dx.dr += gt_dr;

    //// instead, look at the change between pred and gt, and weight t/b/l/r appropriately...
    //// need the real derivative here (I think?)
    //float delta_t = fmax(truth_tblr.top, pred_t) - fmin(truth_tblr.top, pred_t);
    //float delta_b = fmax(truth_tblr.bot, pred_b) - fmin(truth_tblr.bot, pred_b);
    //float delta_l = fmax(truth_tblr.left, pred_l) - fmin(truth_tblr.left, pred_l);
    //float delta_r = fmax(truth_tblr.right, pred_r) - fmin(truth_tblr.right, pred_r);

    //dx.dt *= delta_t / (delta_t + delta_b);
    //dx.db *= delta_b / (delta_t + delta_b);
    //dx.dl *= delta_l / (delta_l + delta_r);
    //dx.dr *= delta_r / (delta_l + delta_r);

//#ifdef DEBUG_PRINTS
    /*printf("  directions dt: ");
    if ((pred_tblr.top < truth_tblr.top && dx.dt > 0) || (pred_tblr.top > truth_tblr.top && dx.dt < 0)) {
      printf("[check]");
    } else {
      printf("X");
    }
    printf(", ");
    if ((pred_tblr.bot < truth_tblr.bot && dx.db > 0) || (pred_tblr.bot > truth_tblr.bot && dx.db < 0)) {
      printf("[check]");
    } else {
      printf("X");
    }
    printf(", ");
    if ((pred_tblr.left < truth_tblr.left && dx.dl > 0) || (pred_tblr.left > truth_tblr.left && dx.dl < 0)) {
      printf("[check]");
    } else {
      printf("X");
    }
    printf(", ");
    if ((pred_tblr.right < truth_tblr.right && dx.dr > 0) || (pred_tblr.right > truth_tblr.right && dx.dr < 0)) {
      printf("[check]");
    } else {
      printf("X");
    }
    printf("\n");

    printf("dx dt:%f", dx.dt);
    printf(", db: %f", dx.db);
    printf(", dl: %f", dx.dl);
    printf(", dr: %f | ", dx.dr);
#endif

#ifdef DEBUG_NAN
    if (isnan(dx.dt)) { printf("dt isnan\n"); }
    if (isnan(dx.db)) { printf("db isnan\n"); }
    if (isnan(dx.dl)) { printf("dl isnan\n"); }
    if (isnan(dx.dr)) { printf("dr isnan\n"); }
#endif

//    // No update if 0 or nan
//    if (dx.dt == 0 || isnan(dx.dt)) { dx.dt = 1; }
//    if (dx.db == 0 || isnan(dx.db)) { dx.db = 1; }
//    if (dx.dl == 0 || isnan(dx.dl)) { dx.dl = 1; }
//    if (dx.dr == 0 || isnan(dx.dr)) { dx.dr = 1; }
//
//#ifdef DEBUG_PRINTS
//    printf("dx dt:%f (t: %f, p: %f)", dx.dt, gt_dt, p_dt);
//    printf(", db: %f (t: %f, p: %f)", dx.db, gt_db, p_db);
//    printf(", dl: %f (t: %f, p: %f)", dx.dl, gt_dl, p_dl);
//    printf(", dr: %f (t: %f, p: %f) | ", dx.dr, gt_dr, p_dr);
//#endif */
    return ddx;
}


// representation from x, y, w, h to top, left, bottom, right
boxabs to_tblr(box a) {
    boxabs tblr = { 0 };
    float t = a.y - (a.h / 2);
    float b = a.y + (a.h / 2);
    float l = a.x - (a.w / 2);
    float r = a.x + (a.w / 2);
    tblr.top = t;
    tblr.bot = b;
    tblr.left = l;
    tblr.right = r;
    return tblr;
}


float box_rmse(box a, box b)
{
    return sqrt(pow(a.x - b.x, 2) +
        pow(a.y - b.y, 2) +
        pow(a.w - b.w, 2) +
        pow(a.h - b.h, 2));
}


// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
float box_diou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float d = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float u = pow(d / c, 0.6);
    float diou_term = u;
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, riou_term: %f\n", c, u, diou_term);
#endif
    return iou - diou_term;
}


// https://github.com/Zzh-tju/DIoU-darknet
// https://arxiv.org/abs/1911.08287
float box_ciou(box a, box b)
{
    boxabs ba = box_c(a, b);
    float w = ba.right - ba.left;
    float h = ba.bot - ba.top;
    float c = w * w + h * h;
    float iou = box_iou(a, b);
    if (c == 0) {
        return iou;
    }
    float u = (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    float d = u / c;
    float ar_gt = b.w / b.h;
    float ar_pred = a.w / a.h;
    float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
    float alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
    float ciou_term = d + alpha * ar_loss;                   //ciou
#ifdef DEBUG_PRINTS
    printf("  c: %f, u: %f, riou_term: %f\n", c, u, ciou_term);
#endif
    return iou - ciou_term;
}