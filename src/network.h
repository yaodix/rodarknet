// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H

#include <stdint.h>
#include "layer.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "image.h"
#include "data.h"
#include "tree.h"

typedef enum {
    CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM
} learning_rate_policy;

typedef struct network{
    float *workspace;
    int n;                          //�����ܲ���
    int batch;                    //һ��batch������ͼƬ�������������subdivision
	int *seen;                     //�Ѿ���ȡ��ͼƬ����
    float epoch;                //ѵ���Ĵ���
    int subdivisions;         // batch/subdivision ��ÿ��ͼƬ������������������batch����ʱ���ٷ��򴫲�
    float momentum;
    float decay;
    layer *layers;               //ָ������Ĳ�
    int outputs;
    float *output;
    learning_rate_policy policy;       //ѧϰ�ʲ���

    float learning_rate;          //ѧϰ��
    float gamma;                   //���ڼ���ѧϰ�ʣ�������0x0102
    float scale;                       //���ڼ���ѧϰ�ʣ�������0x0102
    float power;                    //���ڼ���ѧϰ�ʣ�������0x0102
    int time_steps;
    int step;                           //���ڼ���ѧϰ�ʣ�������0x0102
    int max_batches;
    float *scales;                 //���ڼ���ѧϰ�ʣ�������0x0102
    int   *steps;                    //���ڼ���ѧϰ�ʣ�������0x0102
    int num_steps;              //steps�е����ݸ���
    int burn_in;

    int adam;           //adam �㷨
    float B1;           ////һ�׾ع��Ƶ�ָ��˥����
    float B2;           //���׾ع��Ƶ�ָ��˥����
    float eps;          //Ϊ�˷�ֹ��ʵ���г�����

    int inputs;                 //h*w*c
    int h, w, c;                //����ͼ��ĸߣ���ͨ����
    int max_crop;          ////����ͼƬ���ŵ����ֵ
    int min_crop;           //����ͼƬ���ŵ���Сֵ
    int flip; // horizontal flip 50% probability augmentaiont for classifier training (default = 1)
    float angle;
    float aspect;               //aspects:���÷�λ����������
    float exposure;
    float saturation;
    float hue;
	int small_object;       //???

    int gpu_index;
    tree *hierarchy;

    #ifdef GPU
    float **input_gpu;
    float **truth_gpu;
	float **input16_gpu;
	float **output16_gpu;
	size_t *max_input16_size;
	size_t *max_output16_size;
	int wait_stream;
    #endif
} network;

typedef struct network_state {
    float *truth;       //
    float *input;
    float *delta;
    float *workspace;
    int train;
    int index;
    network net;
} network_state;

#ifdef GPU
float train_networks(network *nets, int n, data d, int interval);
void sync_nets(network *nets, int n, int interval);
float train_network_datum_gpu(network net, float *x, float *y);
float *network_predict_gpu(network net, float *input);
float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float *get_network_output_gpu(network net);
void forward_network_gpu(network net, network_state state);
void backward_network_gpu(network net, network_state state);
void update_network_gpu(network net);
#endif

float get_current_rate(network net);
int get_current_batch(network net);
void free_network(network net);
void compare_networks(network n1, network n2, data d);
char *get_layer_string(LAYER_TYPE a);

network make_network(int n);
void forward_network(network net, network_state state);
void backward_network(network net, network_state state);
void update_network(network net);

float train_network(network net, data d);
float train_network_batch(network net, data d, int n);
float train_network_sgd(network net, data d, int n);
float train_network_datum(network net, float *x, float *y);

matrix network_predict_data(network net, data test);
YOLODLL_API float *network_predict(network net, float *input);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
image get_network_image(network net);
image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
int get_network_input_size(network net);
float get_network_cost(network net);
YOLODLL_API layer* get_network_layer(network* net, int i);
YOLODLL_API detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
YOLODLL_API detection *make_network_boxes(network *net, float thresh, int *num);
YOLODLL_API void free_detections(detection *dets, int n);

YOLODLL_API detectionR *get_network_rboxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
YOLODLL_API detectionR *make_network_rboxes(network *net, float thresh, int *num);
YOLODLL_API void free_detectionsR(detectionR *dets, int n);

YOLODLL_API void reset_rnn(network *net);
YOLODLL_API network *load_network_custom(char *cfg, char *weights, int clear, int batch);
YOLODLL_API network *load_network(char *cfg, char *weights, int clear);
YOLODLL_API float *network_predict_image(network *net, image im);
YOLODLL_API void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show);
YOLODLL_API int network_width(network *net);
YOLODLL_API int network_height(network *net);

YOLODLL_API void optimize_picture(network *net, image orig, int max_layer, float scale, float rate, float thresh, int norm);

int get_network_nuisance(network net);
int get_network_background(network net);
YOLODLL_API void fuse_conv_batchnorm(network net);
YOLODLL_API void calculate_binary_weights(network net);

#ifdef __cplusplus
}
#endif

#endif
