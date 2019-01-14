//
// Created by pesong on 18-9-5.
// used to process the image format
//

#ifndef NCS_UTIL_H
#define NCS_UTIL_H
#define NAME_SIZE 100

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cv_bridge/cv_bridge.h>

#include <mvnc.h>


extern bool g_graph_Success;
extern ncStatus_t retCodeDet;

extern struct ncDeviceHandle_t* deviceHandlePtr;
extern struct ncGraphHandle_t* graphHandlePtr_seg;
extern struct ncGraphHandle_t* graphHandlePtr_det;

extern void* graphFileBuf_seg;
extern void* graphFileBuf_det;

extern unsigned int graphFileLenSeg;
extern unsigned int graphFileLenDet;

// Now we need to allocate graph and create and in/out fifos
extern struct ncFifoHandle_t* inFifoHandlePtr_seg;
extern struct ncFifoHandle_t* outFifoHandlePtr_seg;
extern struct ncFifoHandle_t* inFifoHandlePtr_det;
extern struct ncFifoHandle_t* outFifoHandlePtr_det;

extern int numClasses_;
extern float det_threshold;

// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;

typedef struct{
    int label;
    int x;
    int y;
    int width;
    int height;
    float prob;
} Box;

class image {
public:
    int w;
    int h;
    int c;
    float *data;

    image() {
        data = nullptr;
    };

    image(int _w, int _h, int _c) {
        w = _w;
        h = _h;
        c = _c;
        data = new float[h * w * c]();
    };

    ~image() {
        delete[] data;
    };
};


extern image *ipl_to_image(IplImage *src);
extern void ipl_into_image(IplImage *src, image *im);

extern unsigned char* image_to_stb(image* in);
extern unsigned char* cvMat_to_charImg(cv::Mat pic);
extern void *LoadFile(const char *path, unsigned int *length);
extern float *LoadImage32(unsigned char *img, int target_w, int target_h, int ori_w, int ori_h, float *mean);

extern cv::Mat seg_result_process(float* output, int h, int w);
extern void ssd_result_process(float *output, std::vector<Box> &result, cv::Mat &image, int numClasses_);
extern bool Overlay_on_image(cv::Mat& image, float* object_info, int Length, Box& single_box);
extern void NMS(std::vector <Box> &M);
extern void sizeSort(std::vector <Box> &M);
extern inline float getOverlap(const cv::Rect &b1, const cv::Rect &b2);

extern double getWallTime();

#endif //NCS_UTIL_H
