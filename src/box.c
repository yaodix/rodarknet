#include "box.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
box float_to_box(float *f)
{
    box b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    return b;
}

rbox float_to_rbox(float *f)
{
    rbox b;
    b.x = f[0];
    b.y = f[1];
    b.w = f[2];
    b.h = f[3];
    b.a = f[4];
    return b;
}

dbox derivative(box a, box b)
{
    dbox d;
    d.dx = 0;
    d.dw = 0;
    float l1 = a.x - a.w/2;
    float l2 = b.x - b.w/2;
    if (l1 > l2){
        d.dx -= 1;
        d.dw += .5;
    }
    float r1 = a.x + a.w/2;
    float r2 = b.x + b.w/2;
    if(r1 < r2){
        d.dx += 1;
        d.dw += .5;
    }
    if (l1 > r2) {
        d.dx = -1;
        d.dw = 0;
    }
    if (r1 < l2){
        d.dx = 1;
        d.dw = 0;
    }

    d.dy = 0;
    d.dh = 0;
    float t1 = a.y - a.h/2;
    float t2 = b.y - b.h/2;
    if (t1 > t2){
        d.dy -= 1;
        d.dh += .5;
    }
    float b1 = a.y + a.h/2;
    float b2 = b.y + b.h/2;
    if(b1 < b2){
        d.dy += 1;
        d.dh += .5;
    }
    if (t1 > b2) {
        d.dy = -1;
        d.dh = 0;
    }
    if (b1 < t2){
        d.dy = 1;
        d.dh = 0;
    }
    return d;
}

float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

float box_intersection(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

float box_union(box a, box b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

float box_iou(box a, box b)
{
    return box_intersection(a, b)/box_union(a, b);
}



//*********************旋转矩形IOU计算
//计算IOU
void getRboxVertices(rbox ra, pointf * vertices)
{
    //矩形中心为原点的正矩形顶点坐标
    pointf pts[4];    //左上，右上，右下，左下顺序,x+向右，y+向上，原点为矩形中心
    pts[0].x = -ra.w / 2;
    pts[0].y = ra.h / 2;
    pts[1].x = ra.w / 2;
    pts[1].y = ra.h / 2;

    pts[2].x = ra.w / 2;
    pts[2].y = -ra.h / 2;
    pts[3].x = -ra.w / 2;
    pts[3].y = -ra.h / 2;

    float thelta = ra.a;  //换算后的角度
    float x2 = ra.x;
    float y2 = ra.y;
    for (int i = 0; i < 4; i++)//计算旋转后的角点
    {
        float x1 = pts[i].x;
        float y1 = pts[i].y;
        float x = (x1)*cos( thelta) - (y1)*sin( thelta);
        float y = (x1)*sin( thelta) + (y1)*cos( thelta);

        vertices[i].x = x2 + x;   //坐标系切换，x+向右，y+向下，原点为图像左上角
        vertices[i].y = y2 - y;
    }

}

// 计算交点，(-1,-1)没有交点 
pointf intersectionOf2SegLine(pointf p1, pointf p2, pointf p3, pointf p4)
{
    pointf ptNull,ptRes;
    ptNull.x = -1;
    ptNull.y = -1;

    float x1 = p1.x, x2 = p2.x, x3 = p3.x, x4 = p4.x;
    float y1 = p1.y, y2 = p2.y, y3 = p3.y, y4 = p4.y;
    double denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);
    if (denom == 0.0) { // Lines are parallel.
        return ptNull;
    }
    double ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom;
    double ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / denom;
    if (ua >= 0.0f && ua <= 1.0f && ub >= 0.0f && ub <= 1.0f) {
        // Get the intersection point.
        ptRes.x = (x1 + ua*(x2 - x1));
        ptRes.y = (y1 + ua*(y2 - y1));
        return  ptRes;
    }
    return ptNull;
}

int  isInsideRbox(pointf pt, rbox ra)
{
    pointf ver[4];
    pointf center;
    center.x = ra.x;
    center.y = ra.y;
    getRboxVertices(ra, ver);
    for (int i = 0; i < 4; i++)
    {
        pointf sec = intersectionOf2SegLine(center, pt, ver[i], ver[(i + 1) % 4]); 
        if (sec.x >0)
        {
            return 0;  // 有交点不在矩形内
        }
    }
    return 1;
}

float triangleArea(pointf p1, pointf p2, pointf p3)
{
    float s = ((p3.x - p1.x)*(p2.y - p1.y) - (p2.x - p1.x)*(p3.y - p1.y)) / 2.0;
    return s > 0 ? s : -s;
}

//返回角度
double get2VecAngle(pointf pOrigin, pointf p1, pointf p2)  // 
{
    double dLineDirVec1[3];
    double dLineDirVec2[3];

    dLineDirVec1[0] = p1.x - pOrigin.x;
    dLineDirVec1[1] = p1.y - pOrigin.y;
    dLineDirVec1[2] = 0;

    dLineDirVec2[0] = p2.x - pOrigin.x;
    dLineDirVec2[1] = p2.y - pOrigin.y;
    dLineDirVec2[2] = 0;

    return (acos((dLineDirVec1[0] * dLineDirVec2[0] + dLineDirVec1[1] * dLineDirVec2[1] + dLineDirVec1[2] * dLineDirVec2[2])
        / (sqrt(dLineDirVec1[0] * dLineDirVec1[0] + dLineDirVec1[1] * dLineDirVec1[1] + dLineDirVec1[2] * dLineDirVec1[2])
            *sqrt(dLineDirVec2[0] * dLineDirVec2[0] + dLineDirVec2[1] * dLineDirVec2[1] + dLineDirVec2[2] * dLineDirVec2[2])))) / CV_PI * 180.0;
}
float rbox_intersection(rbox ra, rbox rb)
{
    pointf joint;
    pointf joints[30];
    pointf vertices_ra[4];
    pointf vertices_rb[4];
    int cnt = 0;
    getRboxVertices(ra, vertices_ra);
    getRboxVertices(rb, vertices_rb);

    //获取边的交点
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            joint = intersectionOf2SegLine(vertices_ra[i], vertices_ra[(i + 1) % 4], vertices_rb[j], vertices_rb[(j + 1) % 4]);
            if (joint.x >0)
            {
                joints[cnt++] = joint;
            }
        }
    }
    //查看顶点是否在矩形内
    for (int i = 0; i < 4; i++)
    {
        if (isInsideRbox(vertices_ra[i], rb))
        {
            joints[cnt++] = vertices_ra[i];
        }
    }
    for (int i = 0; i < 4; i++)
    {
        if (isInsideRbox(vertices_rb[i], ra))
        {
            joints[cnt++] = vertices_rb[i];
        }
    }

    //对点进行排序
    pointf joints_center = { 0.,0. }, joint_center_Yminus = {0.,0.};
    float angles[30];
    for (int i = 0; i < cnt; i++)
    {
        joints_center.x += joints[i].x;
        joints_center.y += joints[i].y;
    }
    joints_center.x = joints_center.x / cnt;
    joints_center.y = joints_center.y / cnt;
    joint_center_Yminus.x = joints_center.x;
    joint_center_Yminus.y = joints_center.y - 0.1;

    for (int i = 0; i < cnt; i++)
    {
        double angle_clock = get2VecAngle(joints_center, joint_center_Yminus, joints[i]);
        if (joints[i].x >= joints_center.x)
        {
            angles[i] = angle_clock;
        }
        else
        {
            angles[i] = 360 - angle_clock;
        }
    }

    //排序,从小到大 即顺时针
    for (int i = 0; i < cnt; i++)
    {
        for (int j = i + 1; j < cnt; j++)
        {
            if (angles[i] > angles[j])
            {
                double temp = angles[i];
                angles[i] = angles[j];
                angles[j] = temp;

                pointf tempPt;
                tempPt.x = joints[i].x;
                tempPt.y = joints[i].y;
                joints[i].x = joints[j].x;
                joints[i].y = joints[j].y;
                joints[j].x = tempPt.x;
                joints[j].y = tempPt.y;

            }
        }
    }
    /*
    for (int i = 0; i < cnt; i++)
    {
        circle(img, Point(joints[i].x, joints[i].y), 3, Scalar(0, 250, 5), 5);
        putText(img, to_string(i), Point(joints[i].x, joints[i].y), 1.0, 3, Scalar(0, 250, 5), 3);
    }
    */
    //计算面积
    float areaSum = 0;
    for (int i = 1; i < cnt - 1; i++)
    {
        float area = triangleArea(joints[0], joints[i], joints[i + 1]);
       //// cv::line(img, Point(joints[0].x, joints[0].y), Point(joints[i + 1].x, joints[i + 1].y), Scalar(255, 0, 0), 3);
       // cv::line(img, Point(joints[0].x, joints[0].y), Point(joints[i].x, joints[i].y), Scalar(255, 0, 0), 3);
     //   cv::line(img, Point(joints[i].x, joints[i].y), Point(joints[i + 1].x, joints[i + 1].y), Scalar(255, 0, 0), 3);
        areaSum += area;
    }
    return areaSum;
}

float rbox_union(rbox ra, rbox rb)
{
    float i = rbox_intersection(ra, rb);
    float u = ra.w*ra.h + rb.w*rb.h - i;
    return u;
}

float rbox_iou(rbox ra, rbox rb)
{
    return rbox_intersection(ra, rb) / rbox_union(ra, rb);
}

float box_rmse(box a, box b)
{
    return sqrt(pow(a.x-b.x, 2) + 
                pow(a.y-b.y, 2) + 
                pow(a.w-b.w, 2) + 
                pow(a.h-b.h, 2));
}

dbox dintersect(box a, box b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    dbox dover = derivative(a, b);
    dbox di;

    di.dw = dover.dw*h;
    di.dx = dover.dx*h;
    di.dh = dover.dh*w;
    di.dy = dover.dy*w;

    return di;
}

dbox dunion(box a, box b)
{
    dbox du;

    dbox di = dintersect(a, b);
    du.dw = a.h - di.dw;
    du.dh = a.w - di.dh;
    du.dx = -di.dx;
    du.dy = -di.dy;

    return du;
}


void test_dunion()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dunion(a,b);
    printf("Union: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_union(a, b);
    float xinter = box_union(dxa, b);
    float yinter = box_union(dya, b);
    float winter = box_union(dwa, b);
    float hinter = box_union(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Union Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}
void test_dintersect()
{
    box a = {0, 0, 1, 1};
    box dxa= {0+.0001, 0, 1, 1};
    box dya= {0, 0+.0001, 1, 1};
    box dwa= {0, 0, 1+.0001, 1};
    box dha= {0, 0, 1, 1+.0001};

    box b = {.5, .5, .2, .2};
    dbox di = dintersect(a,b);
    printf("Inter: %f %f %f %f\n", di.dx, di.dy, di.dw, di.dh);
    float inter =  box_intersection(a, b);
    float xinter = box_intersection(dxa, b);
    float yinter = box_intersection(dya, b);
    float winter = box_intersection(dwa, b);
    float hinter = box_intersection(dha, b);
    xinter = (xinter - inter)/(.0001);
    yinter = (yinter - inter)/(.0001);
    winter = (winter - inter)/(.0001);
    hinter = (hinter - inter)/(.0001);
    printf("Inter Manual %f %f %f %f\n", xinter, yinter, winter, hinter);
}

void test_box()
{
    test_dintersect();
    test_dunion();
    box a = {0, 0, 1, 1};
    box dxa= {0+.00001, 0, 1, 1};
    box dya= {0, 0+.00001, 1, 1};
    box dwa= {0, 0, 1+.00001, 1};
    box dha= {0, 0, 1, 1+.00001};

    box b = {.5, 0, .2, .2};

    float iou = box_iou(a,b);
    iou = (1-iou)*(1-iou);
    printf("%f\n", iou);
    dbox d = diou(a, b);
    printf("%f %f %f %f\n", d.dx, d.dy, d.dw, d.dh);

    float xiou = box_iou(dxa, b);
    float yiou = box_iou(dya, b);
    float wiou = box_iou(dwa, b);
    float hiou = box_iou(dha, b);
    xiou = ((1-xiou)*(1-xiou) - iou)/(.00001);
    yiou = ((1-yiou)*(1-yiou) - iou)/(.00001);
    wiou = ((1-wiou)*(1-wiou) - iou)/(.00001);
    hiou = ((1-hiou)*(1-hiou) - iou)/(.00001);
    printf("manual %f %f %f %f\n", xiou, yiou, wiou, hiou);
}

dbox diou(box a, box b)
{
    float u = box_union(a,b);
    float i = box_intersection(a,b);
    dbox di = dintersect(a,b);
    dbox du = dunion(a,b);
    dbox dd = {0,0,0,0};

    if(i <= 0 || 1) {
        dd.dx = b.x - a.x;
        dd.dy = b.y - a.y;
        dd.dw = b.w - a.w;
        dd.dh = b.h - a.h;
        return dd;
    }

    dd.dx = 2*pow((1-(i/u)),1)*(di.dx*u - du.dx*i)/(u*u);
    dd.dy = 2*pow((1-(i/u)),1)*(di.dy*u - du.dy*i)/(u*u);
    dd.dw = 2*pow((1-(i/u)),1)*(di.dw*u - du.dw*i)/(u*u);
    dd.dh = 2*pow((1-(i/u)),1)*(di.dh*u - du.dh*i)/(u*u);
    return dd;
}

typedef struct{
    int index;
    int class_id;
    float **probs;
} sortable_bbox;

int nms_comparator(const void *pa, const void *pb)
{
    sortable_bbox a = *(sortable_bbox *)pa;
    sortable_bbox b = *(sortable_bbox *)pb;
    float diff = a.probs[a.index][b.class_id] - b.probs[b.index][b.class_id];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    sortable_bbox *s = calloc(total, sizeof(sortable_bbox));

    for(i = 0; i < total; ++i){
        s[i].index = i;       
        s[i].class_id = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < classes; ++k){
        for(i = 0; i < total; ++i){
            s[i].class_id = k;
        }
        qsort(s, total, sizeof(sortable_bbox), nms_comparator);
        for(i = 0; i < total; ++i){
            if(probs[s[i].index][k] == 0) continue;
            box a = boxes[s[i].index];
            for(j = i+1; j < total; ++j){
                box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}

int nms_comparator_v3(const void *pa, const void *pb)
{
    detection a = *(detection *)pa;
    detection b = *(detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

int rnms_comparator_v3(const void *pa, const void *pb)
{
    detectionR a = *(detectionR *)pa;
    detectionR b = *(detectionR *)pb;
    float diff = 0;
    if (b.sort_class >= 0) {
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
    }
    else {
        diff = a.objectness - b.objectness;
    }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void do_nms_obj(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (i = 0; i < total; ++i) {
        dets[i].sort_class = -1;
    }

    qsort(dets, total, sizeof(detection), nms_comparator_v3);
    for (i = 0; i < total; ++i) {
        if (dets[i].objectness == 0) continue;
        box a = dets[i].bbox;
        for (j = i + 1; j < total; ++j) {
            if (dets[j].objectness == 0) continue;
            box b = dets[j].bbox;
            if (box_iou(a, b) > thresh) {
                dets[j].objectness = 0;
                for (k = 0; k < classes; ++k) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;
        }
        qsort(dets, total, sizeof(detection), nms_comparator_v3);
        for (i = 0; i < total; ++i) {
            //printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            box a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                box b = dets[j].bbox;
                if (box_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void do_rnms_sort(detectionR *dets, int total, int classes, float thresh)
{
    int i, j, k;
    k = total - 1;
    for (i = 0; i <= k; ++i) {
        if (dets[i].objectness == 0) {
            detectionR swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
        }
    }
    total = k + 1;

    for (k = 0; k < classes; ++k) {
        for (i = 0; i < total; ++i) {
            dets[i].sort_class = k;       //按类进行排序
        }
        qsort(dets, total, sizeof(detectionR), rnms_comparator_v3);

        for (i = 0; i < total; ++i) {
            //printf("  k = %d, \t i = %d \n", k, i);
            if (dets[i].prob[k] == 0) continue;
            rbox a = dets[i].bbox;
            for (j = i + 1; j < total; ++j) {
                rbox b = dets[j].bbox;
                //float angle_diff = fabsf(b.a -a.a);
                if (rbox_iou(a, b) > thresh) {
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

void do_nms(box *boxes, float **probs, int total, int classes, float thresh)
{
    int i, j, k;
    for(i = 0; i < total; ++i){
        int any = 0;
        for(k = 0; k < classes; ++k) any = any || (probs[i][k] > 0);
        if(!any) {
            continue;
        }
        for(j = i+1; j < total; ++j){
            if (box_iou(boxes[i], boxes[j]) > thresh){
                for(k = 0; k < classes; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}



box encode_box(box b, box anchor)
{
    box encode;
    encode.x = (b.x - anchor.x) / anchor.w;
    encode.y = (b.y - anchor.y) / anchor.h;
    encode.w = log2(b.w / anchor.w);
    encode.h = log2(b.h / anchor.h);
    return encode;
}

box decode_box(box b, box anchor)
{
    box decode;
    decode.x = b.x * anchor.w + anchor.x;
    decode.y = b.y * anchor.h + anchor.y;
    decode.w = pow(2., b.w) * anchor.w;
    decode.h = pow(2., b.h) * anchor.h;
    return decode;
}
