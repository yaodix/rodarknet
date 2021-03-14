#ifndef BOX_H
#define BOX_H

#ifdef YOLODLL_EXPORTS
#if defined(_MSC_VER)
#define YOLODLL_API __declspec(dllexport) 
#else
#define YOLODLL_API __attribute__((visibility("default")))
#endif
#else
#if defined(_MSC_VER)
#define YOLODLL_API
#else
#define YOLODLL_API
#endif
#endif

#ifndef CV_PI
#define CV_PI 3.141592653
#endif // !CV_PI

typedef struct pointf
{
    float x, y;
}pointf;

typedef struct{
    float x, y, w, h;
} box;

typedef struct {
    float x, y, w, h, a;    //a ÊÇangle [0,PI)
} rbox;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

typedef struct detection {
	box bbox;
	int classes;
	float *prob;
	float *mask;
	float objectness;
	int sort_class;
} detection;

typedef struct detectionR {
    rbox bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detectionR;

typedef struct detection_with_class {
	detection det;
	// The most probable class id: the best class index in this->prob.
	// Is filled temporary when processing results, otherwise not initialized
	int best_class;
} detection_with_class;

typedef struct detectionR_with_class {
    detectionR det;
    // The most probable class id: the best class index in this->prob.
    // Is filled temporary when processing results, otherwise not initialized
    int best_class;
} detectionR_with_class;

box float_to_box(float *f);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh);
YOLODLL_API void do_nms_sort(detection *dets, int total, int classes, float thresh);

YOLODLL_API void do_nms_obj(detection *dets, int total, int classes, float thresh);
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);
// Creates array of detections with prob > thresh and fills best_class for them
// Return number of selected detections in *selected_detections_num
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num);

//rotated box funcs
rbox float_to_rbox(float *f);
float rbox_iou(rbox a, rbox b);
void getRboxVertices(rbox ra, pointf * vertices);

void do_rnms(box *boxes, float **probs, int total, int classes, float thresh);
YOLODLL_API void do_rnms_sort(detectionR *dets, int total, int classes, float thresh);


#endif
