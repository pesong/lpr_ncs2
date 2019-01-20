#ifndef LICENSE_PLATE_RECOGNITION_ROS_NCSV2_LPR_NCS_HPP
#define LICENSE_PLATE_RECOGNITION_ROS_NCSV2_LPR_NCS_HPP

#include <iostream>
#include <math.h>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <thread>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/algorithm/string.hpp>

// ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

// msgs
#include <lpr_ncs2/BoundingBox.h>
#include <lpr_ncs2/BoundingBoxes.h>

// NCS
#include "ncs_utils/ncs_util.h"
#include <mvnc.h>
#include "SegmentationFreeRecognizer.h"
#include "Pipeline.h"
#include "PlateInfo.h"
#include "opencv2/opencv.hpp"

// Check for xServer
#include <X11/Xlib.h>


namespace lpr_ncs {

    // graph file name
    char *GRAPH_FILE_NAME_DET;
    char *GRAPH_FILE_NAME_FINE;
    //! image dimensions, network mean values for each channel in BGR order.
    int networkDim;
    int target_h;
    int target_w;

    float networkMean[] = {0., 0., 0.};

    //image buffer
    float* imageBufFP32Ptr_fine;
    float* imageBufFP32Ptr_det;

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

        /*!
         * Callback of camera.
         * @param[in] msg image pointer.
         */
        void imageCallback(const sensor_msgs::ImageConstPtr& msg);
        void plates_infer(cv::Mat image_in);
        cv::Mat infer_fine_horizon(cv::Mat image_in, int leftPadding,int rightPadding);
        std::pair<std::string,float> infer_det(cv::Mat Image,std::vector<std::string> mapping_table);


        //! ROS node handle.
        ros::NodeHandle nodeHandle_;

        //! Advertise and subscribe to image topics.
        image_transport::ImageTransport imageTransport_;

        //! ROS subscriber and publisher.
        image_transport::Subscriber imageSubscriber_;
        image_transport::Publisher imageSegPub_;

        float det_threshold;
    };
}




#endif //LICENSE_PLATE_RECOGNITION_ROS_NCSV2_LPR_NCS_HPP
