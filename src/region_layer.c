#include "region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define DOABS 1

region_layer make_region_layer(int batch, int w, int h, int n, int classes, int coords, int max_boxes)
{
    region_layer l = {0};
    l.type = REGION;
    int anchor_valCnts = 3;   // 描述每个anchor的参数个数
    l.n = n;            //anchor的个数，v2中为5 , rot_v2 为6 个
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.classes = classes;
    l.coords = coords;
    l.cost = calloc(1, sizeof(float));              //损失，即loss
    l.biases = calloc(n*anchor_valCnts, sizeof(float));              //anchors的存储位置，一个anchor对应两个值, rot_v2 对应3个值
    l.bias_updates = calloc(n*anchor_valCnts, sizeof(float));                //bias_updates 
    l.outputs = h*w*n*(classes + coords + 1);   //  一张训练图片经过region_layer层后得到的输出元素个数
    l.inputs = l.outputs; // 一张训练图片输入到reigon_layer层的元素个数（注意是一张图片，对于region_layer，输入和输出的元素个数相等）
                          /**
                          * 每张图片含有的真实矩形框参数的个数（max_boxes表示一张图片中最多有max_boxes个ground truth矩形框，
                          * 每个真实矩形框有5个参数，包括x,y,w,h四个定位参数，以及物体类别）,实际上每张图片可能并没有max_boxes个真实矩形框，
                          * 也能没有这么多参数，但为了保持一致性，还是会留着这么大的存储空间，只是其中的值未空而已.                          * 
                          */
    l.max_boxes = max_boxes;
    l.truths = max_boxes*(l.coords + 1);  //  V2 是5即 (l.coords + 1)
    l.delta = calloc(batch*l.outputs, sizeof(float));
    l.output = calloc(batch*l.outputs, sizeof(float));
    int i;
    //for(i = 0; i < n*2; ++i){
    //    l.biases[i] = .5;   //anchors的默认值设为0.5
    //}
    for(i = 0; i < n*anchor_valCnts; i = i+3){
        l.biases[i] = .5;   //anchors的默认值设为0.5
        l.biases[i+1] = .5;   //anchors的默认值设为0.5
        l.biases[i + 2] = CV_PI/12.0+(CV_PI / 6.0)*(i / 3);
    }

    l.forward = forward_region_layer;
    l.backward = backward_region_layer;
#ifdef GPU
    l.forward_gpu = forward_region_layer_gpu;
    l.backward_gpu = backward_region_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "detection\n");
    srand(0);

    return l;
}

void resize_region_layer(layer *l, int w, int h)
{
    int old_w = l->w;
    int old_h = l->h;
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + l->coords + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

#ifdef GPU
    if (old_w < w || old_h < h) {
        cuda_free(l->delta_gpu);
        cuda_free(l->output_gpu);

        l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
        l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
    }
#endif
}

/** 获取某个矩形框的4个定位信息（根据输入的矩形框索引从l.output中获取该矩形框的定位信息x,y,w,h）.
// get bounding box
// x: data pointer of feature map
// biases: data pointer of anchor box data
// biases[2*n] = width of anchor box
// biases[2*n+1] = height of anchor box
// n: output bounding box for each cell in the feature map
// index: output bounding box index in the cell
// i: `cx` in the paper
// j: 'cy' in the paper
// (cx, cy) is the offset from the left top corner of the feature map
// (w, h) is the size of feature map (do normalization in the code)
*/
box get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    box b;
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2*n];
    b.h = exp(x[index + 3]) * biases[2*n+1];
    if(DOABS){
        b.w = exp(x[index + 2]) * biases[2*n]   / w;
        b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    }
    return b;
}

rbox get_region_rbox(float *x, float *biases, int n, int index, int i, int j, int w, int h)
{
    rbox b;
    b.x = (i + logistic_activate(x[index + 0])) / w;           //归一化到1，此处 w=13
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[3 * n];
    b.h = exp(x[index + 3]) * biases[3 * n + 1];        
    b.a = exp(x[index + 4]) * biases[3 * n + 2];             //基于anchor的角度预测
    if (DOABS) {
        b.w = exp(x[index + 2]) * biases[3 * n] / w;
        b.h = exp(x[index + 3]) * biases[3 * n + 1] / h;
    }
    return b;
}
//delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
float delta_region_box(box truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
    box pred = get_region_box(x, biases, n, index, i, j, w, h);
    float iou = box_iou(pred, truth);
    // ground truth of the parameters (tx, ty, tw, th)
    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w / biases[2*n]);
    float th = log(truth.h / biases[2*n + 1]);
    if(DOABS){
        tw = log(truth.w*w / biases[2*n]);
        th = log(truth.h*h / biases[2*n + 1]);
    }

    delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
    delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);
    return iou;
}
//      delta_region_rbox(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
float delta_region_rbox(rbox truth, float *x, float *biases, int n, int index, int i, int j, int w, int h, float *delta, float scale)
{
    float angle_diff_scale = 6.0;
    rbox pred = get_region_rbox(x, biases, n, index, i, j, w, h);
    float iou = rbox_iou(pred, truth);
    float riou = iou*(1.0 - angle_diff_scale*(fabsf(pred.a - truth.a) / CV_PI));   //添加角度对IOU的scale
    // ground truth of the parameters (tx, ty, tw, th)
    float tx = (truth.x*w - i);
    float ty = (truth.y*h - j);
    float tw = log(truth.w / biases[3 * n]);
    float th = log(truth.h / biases[3 * n + 1]);

    if (truth.a - 0.001 <0.0001) truth.a = 0.001;
    if (truth.a - 3.1415 >0.00001) truth.a = 3.1415;
    float ta = log(truth.a / biases[3 * n + 2]);    // truth.a中是[0,PI]  这里truth.a可能为0，而log(0) = inf,导致梯度爆炸

    if (DOABS) {
        tw = log(truth.w*w / biases[3 * n]);
        th = log(truth.h*h / biases[3 * n + 1]);       
    }
    delta[index + 0] = scale * (tx - logistic_activate(x[index + 0])) * logistic_gradient(logistic_activate(x[index + 0]));
    delta[index + 1] = scale * (ty - logistic_activate(x[index + 1])) * logistic_gradient(logistic_activate(x[index + 1]));
    delta[index + 2] = scale * (tw - x[index + 2]);
    delta[index + 3] = scale * (th - x[index + 3]);
    delta[index + 4] = 3.*scale * (ta -x[index + 4]);
    return riou;
}

void delta_region_class(float *output, float *delta, int index, int class_id, int classes, tree *hier, float scale, float *avg_cat, int focal_loss)
{
    int i, n;
    if(hier){
        float pred = 1;
        while(class_id >= 0){
            pred *= output[index + class_id];
            int g = hier->group[class_id];
            int offset = hier->group_offset[g];
            for(i = 0; i < hier->group_size[g]; ++i){
                delta[index + offset + i] = scale * (0 - output[index + offset + i]);
            }
            delta[index + class_id] = scale * (1 - output[index + class_id]);

            class_id = hier->parent[class_id];
        }
        *avg_cat += pred;
    } else {
        // Focal loss
        if (focal_loss) {
            // Focal Loss
            float alpha = 0.5;    // 0.25 or 0.5
            //float gamma = 2;    // hardcoded in many places of the grad-formula

            int ti = index + class_id;
            float pt = output[ti] + 0.000000000000001F;
            // http://fooplot.com/#W3sidHlwZSI6MCwiZXEiOiItKDEteCkqKDIqeCpsb2coeCkreC0xKSIsImNvbG9yIjoiIzAwMDAwMCJ9LHsidHlwZSI6MTAwMH1d
            float grad = -(1 - pt) * (2 * pt*logf(pt) + pt - 1);    // http://blog.csdn.net/linmingan/article/details/77885832
            //float grad = (1 - pt) * (2 * pt*logf(pt) + pt - 1);    // https://github.com/unsky/focal-loss

            for (n = 0; n < classes; ++n) {
                delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);

                delta[index + n] *= alpha*grad;

                if (n == class_id) *avg_cat += output[index + n];
            }
        }
        else {
            // default, yoloV2 默认执行
            for (n = 0; n < classes; ++n) {
                delta[index + n] = scale * (((n == class_id) ? 1 : 0) - output[index + n]);
                if (n == class_id) *avg_cat += output[index + n];
            }
        }
    }
}

float logit(float x)
{
    return log(x/(1.-x));
}

float tisnan(float x)
{
    return (x != x);
}


// batch :当前照片是整个batch中第几张
//location计算得到的n就是中段的偏移数（从第几个中段开始，对应是第几个矩形框）
//entry就是小段的偏移数（从几个小段开始，对应具体是那种参数，x,c还是C1），，
//而loc则是最后的定位,前面确定好第几大段中的第几中段中的第几小段的首地址，loc就是从该首地址往后数loc个元素，
//  得到最终定位某个具体参数（x或c或C1）的索引值 。
static int entry_index(layer l, int batch, int location, int entry)
{
    int n = location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(l.coords + l.classes + 1) + entry*l.w*l.h + loc;
}

void softmax_tree(float *input, int batch, int inputs, float temp, tree *hierarchy, float *output);
void forward_region_layer(const region_layer l, network_state state)
{
    int i,j,b,t,n;
    int size = l.coords + l.classes + 1;
    memcpy(l.output, state.input, l.outputs*l.batch*sizeof(float));
    #ifndef GPU
    flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
    #endif
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.h*l.w*l.n; ++i){
            int index = size*i + b*l.outputs;
            //l.output[index + 4] = logistic_activate(l.output[index + 4]);     // 置信概率
            l.output[index + 5] = logistic_activate(l.output[index + 5]);     // 置信概率 rot
        }
    }


#ifndef GPU
    if (l.softmax_tree){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                softmax_tree(l.output + index + 5, 1, 0, 1, l.softmax_tree, l.output + index + 5);
            }
        }
    } else if (l.softmax){
        for (b = 0; b < l.batch; ++b){
            for(i = 0; i < l.h*l.w*l.n; ++i){
                int index = size*i + b*l.outputs;
                //  softmax(l.output + index + 5, l.classes, 1, l.output + index + 5, 1);
                softmax(l.output + index + 6, l.classes, 1, l.output + index + 6, 1);   //rot 
            }
        }
    }
#endif
    if(!state.train) return;
    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));    //梯度清零
    float avg_iou = 0;
    float angle_thresh = CV_PI/12. ;  //角度筛选阈值
    float recall = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0; //一张训练图片所有预测矩形框的平均置信度（矩形框中含有物体的概率），该参数没有实际用处，仅用于输出打印
    int count = 0;              //该batch内检测的target数
    int class_count = 0;
    *(l.cost) = 0;              //损失

    for (b = 0; b < l.batch; ++b) {         //遍历batch内数据
        if(l.softmax_tree){     //不执行，yolo9000用 
            int onlyclass_id = 0;
            for(t = 0; t < l.max_boxes; ++t){    //max_boxes 是设定的最大的ground truth的个数
                box truth = float_to_box(state.truth + t*5 + b*l.truths);   //state.truth 的布局：
                if(!truth.x) break; // continue;
                int class_id = state.truth[t*5 + b*l.truths + 4];
                float maxp = 0;
                int maxi = 0;
                if(truth.x > 100000 && truth.y > 100000){
                    for(n = 0; n < l.n*l.w*l.h; ++n){
                        int index = size*n + b*l.outputs + 5;
                        float scale =  l.output[index-1];
                        float p = scale*get_hierarchy_probability(l.output + index, l.softmax_tree, class_id);
                        if(p > maxp){
                            maxp = p;
                            maxi = n;
                        }
                    }
                    int index = size*maxi + b*l.outputs + 5;
                    delta_region_class(l.output, l.delta, index, class_id, l.classes, l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);
                    ++class_count;
                    onlyclass_id = 1;
                    break;
                }
            }
            if(onlyclass_id) continue;
        }  // V2 不执行 ，yolo9000用 

        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;     //获取预测的box的索引
                    //box pred = get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);    
                    rbox pred = get_region_rbox(l.output, l.biases, n, index, i, j, l.w, l.h);    // 在cell(i, j) 上相对于anchor n的预测结果,
                    float best_iou = 0;
                    float best_angle_diff = 0;
                    int best_class_id = -1;
                    for(t = 0; t < l.max_boxes; ++t){   //默认一张图片中最多就打了max_boxes个物体的标签，
                        //box truth = float_to_box(state.truth + t*5 + b*l.truths);  //state.truth : 存储了网络一个batch图片的真实矩形框信息  ,l.truths = max_boxes*5
                        //int class_id = state.truth[t * 5 + b*l.truths + 4];
                        rbox truth = float_to_rbox(state.truth + t * (l.coords+1) + b*l.truths);
                        int class_id = state.truth[t * (l.coords + 1) + b*l.truths + 5];     //这里需要确定truth的结构 ?

                        if (class_id >= l.classes) continue; // if label contains class_id more than number of classes in the cfg-file
                        if(!truth.x) break; // continue;
                        
                        float iou = rbox_iou(pred, truth);
                        float angle_diff = fabsf(pred.a - truth.a);
                        if (( angle_diff < angle_thresh) &&  (iou > best_iou)   ) {     //找到当前预测box与所有真实标注框的最大IOU
                            best_class_id = state.truth[t * 6 + b*l.truths + 5];     
                            best_iou = iou;
                            best_angle_diff = angle_diff;
                        }
                    }
                    avg_anyobj += l.output[index + 5];
                    //不负责预测物体的anchor的confidece损失
                    l.delta[index + 5] = l.noobject_scale * ((0 - l.output[index + 5]) * logistic_gradient(l.output[index + 5]));  
                    if(l.classfix == -1) l.delta[index + 4] = l.noobject_scale * ((best_iou - l.output[index + 4]) * logistic_gradient(l.output[index + 4]));
                    else{
                        if ( (best_iou > l.thresh)  && ( best_angle_diff < angle_thresh)) {  //best iou > thresh, 我们认为这个bounding box有了对应的groundtruth，把置信度梯度直接设置为0即可
                            l.delta[index + 5] = 0;
                            if(l.classfix > 0){
                                delta_region_class(l.output, l.delta, index + 6, best_class_id, l.classes, l.softmax_tree, l.class_scale*(l.classfix == 2 ? l.output[index + 5] : 1), &avg_cat, l.focal_loss);
                                ++class_count;
                            }
                        }
                    }
                    //不负责预测物体的anchor的xywha损失，迭代初期计算( < 12800),
                    //让所有anchor的预测都接近anchor自身的xywh，这样当它有物体落入这个anchor的时候，anchor的预测
                    //不至于和目标差别太大，相应的损失也会比较小，训练起来会更加容易。
                    if(*(state.net.seen) < 12800){
                        rbox truth = {0};   //当前cell为中心对应的第n个anchor的box
                        truth.x = (i + .5)/l.w;     //cell 的中点 对应tx=0.5
                        truth.y = (j + .5)/l.h;     //对应ty = 0.5
                        truth.w = l.biases[3*n];        //相对于feature map的大小 
                        truth.h = l.biases[3*n+1];      
                        truth.a = l.biases[3*n + 2];
                        if(DOABS){
                            truth.w = l.biases[3*n]/l.w;
                            truth.h = l.biases[3*n+1]/l.h;   //truth的 x,y w,h 都是相对值
                        }
                        delta_region_rbox(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, .01);
                    }
                }
            }
        }//->第一个循环,下面是的二个循环

        for(t = 0; t < l.max_boxes; ++t){
            rbox truth = float_to_rbox(state.truth + t*6 + b*l.truths);
            int class_id = state.truth[t * 6 + b*l.truths + 5];
            if (class_id >= l.classes) {
                printf(" Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] \n", class_id, l.classes, l.classes-1);
                getchar();
                continue; // if label contains class_id more than number of classes in the cfg-file
            }
            if(!truth.x) break; // continue;
            float best_iou = 0;
            float best_angle_diff = 0;
            int best_index = 0;
            int best_n = 0;
            i = (truth.x * l.w);        // 类型的强制转换，计算该truth所在的cell的i,j坐标
            j = (truth.y * l.h);
            //printf("%d %f %d %f\n", i, truth.x*l.w, j, truth.y*l.h);
            rbox truth_shift = truth;
            truth_shift.x = 0;
            truth_shift.y = 0;
            //printf("index %d %d\n",i, j);
            for(n = 0; n < l.n; ++n){   //遍历对应的cell预测出的n个结果，该预测结果基于anchor值来计算
                                                      // 即通过该cell对应的anchors的结果与truth的iou来判断使用哪一个anchor产生的predict来回归
                int index = size*(j*l.w*l.n + i*l.n + n) + b*l.outputs;            //  预测结果中box的索引
                rbox pred = get_region_rbox(l.output, l.biases, n, index, i, j, l.w, l.h);  //获取box尺寸
                //下面这几句是将truth与anchor中心对齐后，计算anchor与truch的iou
                if(l.bias_match){   
                    pred.w = l.biases[3*n];
                    pred.h = l.biases[3*n+1];
                    pred.a = l.biases[3 * n + 2];  
                    if(DOABS){
                        pred.w = l.biases[3*n]/l.w;
                        pred.h = l.biases[3*n+1]/l.h;
                    }
                }
               // printf("pred %d : (%f, %f) %f x %f  angle = %f\n", n ,pred.x, pred.y, pred.w, pred.h,pred.a);
                pred.x = 0;
                pred.y = 0;
                float iou = rbox_iou(pred, truth_shift);
                float angle_diff = fabsf(pred.a - truth.a);
                if (iou > best_iou && angle_diff < angle_thresh ){
                    best_index = index;
                    best_angle_diff = angle_diff;
                    best_iou = iou;
                    best_n = n;    //最优iou对应的anchor索引，然后使用该anchor预测的predict box计算与真实box的误差
                }
            }
           //printf("truth:%d %f (%f, %f) %f x %f  angle =%f \n", best_n, best_iou, truth.x, truth.y, truth.w, truth.h,truth.a);

            float iou = delta_region_rbox(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
            if(iou > .55 && best_angle_diff < angle_thresh) recall += 1;     // 如果iou> 0.5, 认为找到该目标，召回数+1
            avg_iou += iou;

            //l.delta[best_index + 4] = iou - l.output[best_index + 4];
            avg_obj += l.output[best_index + 5];     //对应predict预测的confidence
            l.delta[best_index + 5] = l.object_scale * (1 - l.output[best_index + 5]) * logistic_gradient(l.output[best_index + 5]); //有目标时的损失
            if (l.rescore) {   //定义了rescore表示同时对confidence score进行回归
                l.delta[best_index +5] = l.object_scale * (iou - l.output[best_index + 5]) * logistic_gradient(l.output[best_index +5]);
            }

            if (l.map) class_id = l.map[class_id];
            //类别损失
            delta_region_class(l.output, l.delta, best_index + 6, class_id, l.classes, l.softmax_tree, l.class_scale, &avg_cat, l.focal_loss);  //rot
            ++count;
            ++class_count;
        }
    }
    //printf("\n");
    #ifndef GPU
    flatten(l.delta, l.w*l.h, size*l.n, l.batch, 0);
    #endif
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d\n", avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, count);
}

void backward_region_layer(const region_layer l, network_state state)
{
    axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}

void get_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map)
{
    int i,j,n;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i){
        int row = i / l.w;
        int col = i % l.w;
        for(n = 0; n < l.n; ++n){
            int index = i*l.n + n;
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];
            if(l.classfix == -1 && scale < .5) scale = 0;
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;

            int class_index = index * (l.classes + 5) + 5;
            if(l.softmax_tree){

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if(map){
                    for(j = 0; j < 200; ++j){
                        float prob = scale*predictions[class_index+map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                } else {
                    for(j = l.classes - 1; j >= 0; --j){
                        if(!found && predictions[class_index + j] > .5){
                            found = 1;
                        } else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index+j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            } else {
                for(j = 0; j < l.classes; ++j){
                    float prob = scale*predictions[class_index+j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if(only_objectness){
                probs[index][0] = scale;
            }
        }
    }
}

void get_region_rboxes(layer l, int w, int h, float thresh, float **probs, rbox *boxes, int only_objectness, int *map)
{
    int i, j, n;
    float *predictions = l.output;
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;    //预测结果上的第几个cell
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int index = i*l.n + n;    //
            int p_index = index * (l.classes + 6) + 5;
            float scale = predictions[p_index];
            if (l.classfix == -1 && scale < .5) scale = 0;
            int box_index = index * (l.classes + 6);
            boxes[index] = get_region_rbox(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;
         
            int class_index = index * (l.classes + 6) + 6;
            if (l.softmax_tree) {

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);
                int found = 0;
                if (map) {
                    for (j = 0; j < 200; ++j) {
                        float prob = scale*predictions[class_index + map[j]];
                        probs[index][j] = (prob > thresh) ? prob : 0;
                    }
                }
                else {
                    for (j = l.classes - 1; j >= 0; --j) {
                        if (!found && predictions[class_index + j] > .5) {
                            found = 1;
                        }
                        else {
                            predictions[class_index + j] = 0;
                        }
                        float prob = predictions[class_index + j];
                        probs[index][j] = (scale > thresh) ? prob : 0;
                    }
                }
            }
            else {
                for (j = 0; j < l.classes; ++j) {
                    float prob = scale*predictions[class_index + j];
                    probs[index][j] = (prob > thresh) ? prob : 0;
                }
            }
            if (only_objectness) {
                probs[index][0] = scale;
            }
        }
    }
}

#ifdef GPU

void forward_region_layer_gpu(const region_layer l, network_state state)
{
    /*
       if(!state.train){
       copy_ongpu(l.batch*l.inputs, state.input, 1, l.output_gpu, 1);
       return;
       }
     */
    flatten_ongpu(state.input, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 1, l.output_gpu);
    if(l.softmax_tree){
        int i;
        int count = 5;
        for (i = 0; i < l.softmax_tree->groups; ++i) {
            int group_size = l.softmax_tree->group_size[i];
            softmax_gpu(l.output_gpu+count, group_size, l.classes + 5, l.w*l.h*l.n*l.batch, 1, l.output_gpu + count);
            count += group_size;
        }
    }else if (l.softmax){
        softmax_gpu(l.output_gpu+6, l.classes, l.classes + 6, l.w*l.h*l.n*l.batch, 1, l.output_gpu + 6);   //
    }

    float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
    float *truth_cpu = 0;
    if(state.truth){
        int num_truth = l.batch*l.truths;
        truth_cpu = calloc(num_truth, sizeof(float));
        cuda_pull_array(state.truth, truth_cpu, num_truth);
    }
    cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
    //cudaStreamSynchronize(get_cuda_stream());
    network_state cpu_state = state;
    cpu_state.train = state.train;
    cpu_state.truth = truth_cpu;
    cpu_state.input = in_cpu;
    forward_region_layer(l, cpu_state);
    //cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
    free(cpu_state.input);
    if(!state.train) return;
    cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
    //cudaStreamSynchronize(get_cuda_stream());
    if(cpu_state.truth) free(cpu_state.truth);
}

void backward_region_layer_gpu(region_layer l, network_state state)
{
    flatten_ongpu(l.delta_gpu, l.h*l.w, l.n*(l.coords + l.classes + 1), l.batch, 0, state.delta);
}
#endif


void correct_region_boxes(detection *dets, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    }
    else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (i = 0; i < n; ++i) {
        box b = dets[i].bbox;
        b.x = (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw);
        b.y = (b.y - (neth - new_h) / 2. / neth) / ((float)new_h / neth);
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        dets[i].bbox = b;
    }
}


void get_region_detections(layer l, int w, int h, int netw, int neth, float thresh, int *map, float tree_thresh, int relative, detection *dets)
{
    int i, j, n, z;
    float *predictions = l.output;
    if (l.batch == 2) {
        float *flip = l.output + l.outputs;
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w / 2; ++i) {
                for (n = 0; n < l.n; ++n) {
                    for (z = 0; z < l.classes + l.coords + 1; ++z) {
                        int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                        int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                        float swap = flip[i1];
                        flip[i1] = flip[i2];
                        flip[i2] = swap;
                        if (z == 0) {
                            flip[i1] = -flip[i1];
                            flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
        for (i = 0; i < l.outputs; ++i) {
            l.output[i] = (l.output[i] + flip[i]) / 2.;
        }
    }
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int index = n*l.w*l.h + i;
            for (j = 0; j < l.classes; ++j) {
                dets[index].prob[j] = 0;
            }
            int obj_index = entry_index(l, 0, n*l.w*l.h + i, l.coords);
            int box_index = entry_index(l, 0, n*l.w*l.h + i, 0);
            int mask_index = entry_index(l, 0, n*l.w*l.h + i, 4);
            float scale = l.background ? 1 : predictions[obj_index];
            dets[index].bbox = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);// , l.w*l.h);
            dets[index].objectness = scale > thresh ? scale : 0;
            if (dets[index].mask) {
                for (j = 0; j < l.coords - 4; ++j) {
                    dets[index].mask[j] = l.output[mask_index + j*l.w*l.h];
                }
            }

            int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + !l.background);
            if (l.softmax_tree) {

                hierarchy_predictions(predictions + class_index, l.classes, l.softmax_tree, 0);// , l.w*l.h);
                if (map) {
                    for (j = 0; j < 200; ++j) {
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + map[j]);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
                else {
                    int j = hierarchy_top_prediction(predictions + class_index, l.softmax_tree, tree_thresh, l.w*l.h);
                    dets[index].prob[j] = (scale > thresh) ? scale : 0;
                }
            }
            else {
                if (dets[index].objectness) {
                    for (j = 0; j < l.classes; ++j) {
                        int class_index = entry_index(l, 0, n*l.w*l.h + i, l.coords + 1 + j);
                        float prob = scale*predictions[class_index];
                        dets[index].prob[j] = (prob > thresh) ? prob : 0;
                    }
                }
            }
        }
    }
    correct_region_boxes(dets, l.w*l.h*l.n, w, h, netw, neth, relative);
}
