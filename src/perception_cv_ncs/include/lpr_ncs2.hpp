//
// Created by pesong on 18-9-5.
//

#ifndef PERCEPTION_CV_H
#define PERCEPTION_CV_H

#include <iostream>
#include <math.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <thread>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>

// ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

// msgs
#include <lpr_ncs2/BoundingBox.h>
#include <lpr_ncs2/BoundingBoxes.h>

// NCS
#include "ncs_util.h"
#include <mvnc.h>


namespace lpr_ncs2 {

    // graph file name
    char *GRAPH_FILE_NAME_DET;
    char *GRAPH_FILE_NAME_SEG;
    //! image dimensions, network mean values for each channel in BGR order.
    int networkDim;
    int target_h;
    int target_w;
    //cityscapes:
//    float networkMean[] = {71.60167789, 82.09696889, 72.30608881};
     float networkMean[] = {125., 125., 125.};

    //image buffer
    float* imageBufFP32Ptr;
    cv::Mat ROS_img_resized;

    class LPR_NCS {
    public:

        /*!
        * Constructor.
        */
        explicit LPR_NCS(ros::NodeHandle nh);

        /*!
         * Destructor.
         */
        ~LPR_NCS();

    private:

        /*!
         * Initialize the movidius and ROS connections.
         */
        void init();
        void init_ncs();
        void init_lpr();


        /*!
         * Callback of camera.
         * @param[in] msg image pointer.
         */
        void imageCallback(const sensor_msgs::ImageConstPtr& msg);

        cv::Mat ncs_result_process(float* output, int h, int w);

        cv::Mat camImageCopy_;
        std::thread inferThread_;

        bool imageStatus_ = false;
        bool isNodeRunning_ = true;

        void *segThread();
        void *detThread();
        void *publishThread();

        //! Class labels.
        int numClasses_;
        std::vector <std::string> classLabels_;

        //! thread
        float ssd_threshold;

        void infer();
        cv::Mat getCVImage();

        bool getImageStatus(void);
        bool isNodeRunning(void);

        bool flipFlag;
        double demoTime_;
        int demoDone_ = 0;
        float fps_ = 0;

        cv::Mat seg_out_img;

        //! ROS node handle.
        ros::NodeHandle nodeHandle_;

        //! Advertise and subscribe to image topics.
        image_transport::ImageTransport imageTransport_;

        //! ROS subscriber and publisher.
        image_transport::Subscriber imageSubscriber_;
        image_transport::Publisher imageSegPub_;


    };
}
#endif //PERCEPTION_CV_H
