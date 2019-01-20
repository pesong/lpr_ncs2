#include "lpr_ncs.hpp"


namespace lpr_ncs {

    LPR_NCS::LPR_NCS(ros::NodeHandle nh)
    : nodeHandle_(nh),
      imageTransport_(nodeHandle_){

        ROS_INFO("[Detector] Node started.");

        init_ncs();
        init();
    }

    LPR_NCS::~LPR_NCS() {
        //clear and close ncs
        ROS_INFO("Delete movidius SSD graph");
        ncFifoDestroy(&inFifoHandlePtr_det);
        ncFifoDestroy(&outFifoHandlePtr_det);
        ncGraphDestroy(&graphHandlePtr_det);
        ncDeviceClose(deviceHandlePtr);
        ncDeviceDestroy(&deviceHandlePtr);
    }

    pr::PipelinePR prc("/dl/ros/License_Plate_Recognition_ros_ncsv2/src/perception_cv_ncs/model/cascade.xml",
                       "/dl/ros/License_Plate_Recognition_ros_ncsv2/src/perception_cv_ncs/model/HorizonalFinemapping.prototxt",
                       "/dl/ros/License_Plate_Recognition_ros_ncsv2/src/perception_cv_ncs/model/HorizonalFinemapping.caffemodel",
                       "/dl/ros/License_Plate_Recognition_ros_ncsv2/src/perception_cv_ncs/model/SegmenationFree-Inception.prototxt",
                       "/dl/ros/License_Plate_Recognition_ros_ncsv2/src/perception_cv_ncs/model/SegmenationFree-Inception.caffemodel"
    );
    cv::Mat frame;
    cv::VideoCapture cap;

    //  init movidius:open ncs device, creat graph and read in a graph
    void LPR_NCS::init_ncs() {

        // load param
        bool flip_flag;
        std::string graphPath;
        std::string graphModelDet;
        std::string graphModelFine;

        GRAPH_FILE_NAME_DET = new char[graphPath.length() + 1];
        GRAPH_FILE_NAME_FINE = new char[graphPath.length() + 1];

        nodeHandle_.param("backbone_graph/graph_file/det_graph_name", graphModelDet, std::string("lpr_ncs_v2.graph"));
        nodeHandle_.param("backbone_graph/graph_file/horizon_fine_graph_name", graphModelFine, std::string("horizon_fine_v2.graph"));
        nodeHandle_.param("graph_path", graphPath, std::string("graph"));
        nodeHandle_.param("backbone_graph/networkDim", networkDim, 300);
        nodeHandle_.param("backbone_graph/target_h", target_h, 300);
        nodeHandle_.param("backbone_graph/target_w", target_w, 300);
        nodeHandle_.param("camera/image_flip", flip_flag, false);
        nodeHandle_.param("backbone_graph/threshold", det_threshold, (float) 0.9);


        strcpy(GRAPH_FILE_NAME_DET, (graphPath + "/" + graphModelDet).c_str());
        strcpy(GRAPH_FILE_NAME_FINE, (graphPath + "/" + graphModelFine).c_str());

        // Try to create the first Neural Compute device (at index zero)
        retCodeDet = ncDeviceCreate(0, &deviceHandlePtr);
        if (retCodeDet != NC_OK)
        {   // failed to create the device.
            printf("Could not create NC device\n");
            exit(-1);
        }

        // deviceHandle is created and ready to be opened
        retCodeDet = ncDeviceOpen(deviceHandlePtr);
        if (retCodeDet != NC_OK)
        {   // failed to open the device.
            printf("Could not open NC device\n");
            exit(-1);
        }

        // The device is open and ready to be used.
        printf("Successfully opened NC device!\n");

        // Create the graph
        retCodeFine = ncGraphCreate("refine Graph", &graphHandlePtr_fine);
        retCodeDet = ncGraphCreate("lpr Detection Graph", &graphHandlePtr_det);

        if (retCodeDet != NC_OK)
        {   // error allocating graph
            printf("Could not create graph.\n");
            printf("Error from ncGraphCreate is: %d\n", retCodeDet);
        }else { // successfully created graph.  Now we need to destory it when finished with it.
            // Now we need to allocate graph and create and in/out fifos
            inFifoHandlePtr_fine = NULL;
            outFifoHandlePtr_fine = NULL;
            inFifoHandlePtr_det = NULL;
            outFifoHandlePtr_det = NULL;

            // Now read in a graph file from disk to memory buffer and
            // then allocate the graph based on the file we read
            void* graphFileBuf_fine = LoadFile(GRAPH_FILE_NAME_FINE, &graphFileLenFine);
            retCodeFine = ncGraphAllocateWithFifos(deviceHandlePtr, graphHandlePtr_fine, graphFileBuf_fine, graphFileLenFine, &inFifoHandlePtr_fine, &outFifoHandlePtr_fine);

            void* graphFileBuf_det = LoadFile(GRAPH_FILE_NAME_DET, &graphFileLenDet);
            retCodeDet = ncGraphAllocateWithFifos(deviceHandlePtr, graphHandlePtr_det, graphFileBuf_det, graphFileLenDet, &inFifoHandlePtr_det, &outFifoHandlePtr_det);

            free(graphFileBuf_fine);
            free(graphFileBuf_det);

            if (retCodeFine != NC_OK)
            {   // error allocating graph or fifos
                printf("Could not allocate fine graph with fifos.\n");
                printf("Error from ncGraphAllocateWithFifos is: %d\n", retCodeFine);
            }else{
                // Now graphHandle is ready to go we it can now process inferences.
                printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME_FINE);
            }

            if (retCodeDet != NC_OK)
            {   // error allocating graph or fifos
                printf("Could not allocate detection graph with fifos.\n");
                printf("Error from ncGraphAllocateWithFifos is: %d\n", retCodeDet);
            }else{
                // Now graphHandle is ready to go we it can now process inferences.
                printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME_DET);
            }
        }
    }

    // init ros node
    void LPR_NCS::init() {
        ROS_INFO("[LPR_NCS] init().");

        // load param
        std::string cameraTopicName;
        std::string OutTopicName;
        nodeHandle_.param("subscribers/camera_reading/topic", cameraTopicName, std::string("/camera/image"));
        nodeHandle_.param("subscribers/seg_image/topic", OutTopicName, std::string("/camera/lpr_out"));
        // infer thread
        imageSubscriber_ = imageTransport_.subscribe(cameraTopicName, 1, &LPR_NCS::imageCallback, this);
        imageSegPub_ = imageTransport_.advertise(OutTopicName, 1);

        //.. tmp
//        cap.open(0);
        cap.open("/media/pesong/e/dl_gaussian/model/HyperLPR/Prj-Linux/detect.mp4");
        if(!cap.isOpened()){
            exit(-1);
        }

    }



    // imageCallback
    void LPR_NCS::imageCallback(const sensor_msgs::ImageConstPtr &msg)
    {
        ROS_INFO("[infer:callback] image received.");
//        cv_bridge::CvImagePtr cam_image;
//
//        try {
//            cam_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
//        } catch (cv_bridge::Exception &e) {
//            ROS_ERROR("cv_bridge exception: %s", e.what());
//            return;
//        }
//
//        if(cam_image) {
//            cv::Mat image0 = cam_image->image.clone();
//            printf("begin plate infer");
//            plates_infer(image0);
//        }
        cap.read(frame);
        plates_infer(frame);
    }

    void LPR_NCS::plates_infer(cv::Mat image_in){
//        std::vector<pr:: PlateInfo> res = prc.RunPiplineAsImage(image_in);
        std::vector<pr::PlateInfo> results;
        std::vector<pr::PlateInfo> plates;
        prc.plateDetection->plateDetectionRough(image_in,plates,36,700);
        const int HorizontalPadding = 4;
        cv::Mat out_frame;

        for (pr::PlateInfo plateinfo:plates) {

            cv::Mat image_finemapping = plateinfo.getPlateImage();
            image_finemapping = prc.fineMapping->FineMappingVertical(image_finemapping);
            image_finemapping = pr::fastdeskew(image_finemapping, 5);

            //Segmentation-free
//            image_finemapping = prc.fineMapping->FineMappingHorizon(image_finemapping, 4, HorizontalPadding+3);

            // movidius infer: fine horizon
            image_finemapping = infer_fine_horizon(image_finemapping, 4, HorizontalPadding+3);

//            cv::imshow("fine",image_finemapping);
//            cv::waitKey(1);

            cv::resize(image_finemapping, image_finemapping, cv::Size(136+HorizontalPadding, 36));
            plateinfo.setPlateImage(image_finemapping);

            std::pair<std::string,float> res = prc.segmentationFreeRecognizer->SegmentationFreeForSinglePlate(plateinfo.getPlateImage(),pr::CH_PLATE_CODE);

//            std::pair<std::string,float> res = infer_det(plateinfo.getPlateImage(),pr::CH_PLATE_CODE);

            plateinfo.confidence = res.second;
            plateinfo.setPlateName(res.first);

            results.push_back(plateinfo);
        }

        show_lpr_result(image_in, results, det_threshold, out_frame);
        sensor_msgs::ImagePtr msg_lpr = cv_bridge::CvImage(std_msgs::Header(), "bgr8", out_frame).toImageMsg();
        imageSegPub_.publish(msg_lpr);

    }



    cv::Mat LPR_NCS::infer_fine_horizon(cv::Mat image_in, int leftPadding, int rightPadding){
        cv::Mat cropped;
//        if(FinedVertical.channels()==1)
//            cv::cvtColor(FinedVertical,FinedVertical,cv::COLOR_GRAY2BGR);
        ////prepare img need by movidius
//        cv::resize(image_in, ROS_img_resized_rough, cv::Size(66, 16), 0, 0, CV_INTER_LINEAR);
        unsigned char *img = cvMat_to_charImg(image_in);

        ////infer the refine horizon model
        unsigned int tensorSizeFine = 0;  /* size of image buffer should be: sizeof(float) * reqsize * reqsize * 3;*/
        imageBufFP32Ptr_fine = LoadImage32(img, 66, 16, image_in.cols, image_in.rows, networkMean, &tensorSizeFine);

        // queue the inference to start, when its done the result will be placed on the output fifo
        retCodeFine = ncGraphQueueInferenceWithFifoElem(
                graphHandlePtr_fine, inFifoHandlePtr_fine, outFifoHandlePtr_fine, imageBufFP32Ptr_fine, &tensorSizeFine, NULL);

        if (retCodeFine != NC_OK)
        {   // error queuing input tensor for inference
            printf("Could not queue detection inference\n");
            printf("Error from ncGraphQueueInferenceWithFifoElem is: %d\n", retCodeFine);
            exit(-1);
        }
        else
        {
            // the inference has been started, now read the output queue for the inference result
//            printf("---------------Successfully queued the horizon fine inference for image-----------\n");

            unsigned int outFifoElemSize = 0;
            unsigned int optionSize = sizeof(outFifoElemSize);
            ncFifoGetOption(outFifoHandlePtr_fine,  NC_RO_FIFO_ELEMENT_DATA_SIZE, &outFifoElemSize, &optionSize);

            float* resultDataFP32Ptr = (float*) malloc(outFifoElemSize);
            void* UserParamPtr = NULL;

            // read the output of the inference.  this will be in FP32 since that is how the fifos are created by default.
            retCodeFine = ncFifoReadElem(outFifoHandlePtr_fine, (void*)resultDataFP32Ptr, &outFifoElemSize, &UserParamPtr);
            if (retCodeFine == NC_OK)
            {   // Successfully got the inference result.
                // The inference result is in the buffer pointed to by resultDataFP32Ptr
                int front = static_cast<int>(resultDataFP32Ptr[0] * 127.5);
                int back = static_cast<int>(resultDataFP32Ptr[1] * 127.5);

//                std::cout<< "front: " << front << " back: " << back << std::endl;

                front -= leftPadding ;
                if(front<0) front = 0;
                back +=rightPadding;
                if(back>image_in.cols-1) back=image_in.cols - 1;
                cropped  = image_in.colRange(front,back).clone();
            }
            delete imageBufFP32Ptr_fine;
            free((void*)resultDataFP32Ptr);
        }

        return  cropped;
    }


    std::pair<std::string,float> LPR_NCS::infer_det(cv::Mat Image,std::vector<std::string> mapping_table){

        std::pair<std::string,float> result_;
        cv::transpose(Image,Image);
        cv::Mat code_table;

        ////prepare img need by movidius
//        cv::resize(image_in, ROS_img_resized_rough, cv::Size(66, 16), 0, 0, CV_INTER_LINEAR);
        unsigned char *img = cvMat_to_charImg(Image);
        unsigned int tensorSizeDet = 0;
        imageBufFP32Ptr_det = LoadImage32(img, 160, 40, Image.cols, Image.rows, networkMean, &tensorSizeDet);

        // queue the inference to start, when its done the result will be placed on the output fifo
        retCodeDet = ncGraphQueueInferenceWithFifoElem(
                graphHandlePtr_det, inFifoHandlePtr_det, outFifoHandlePtr_det, imageBufFP32Ptr_det, &tensorSizeDet, NULL);

        if (retCodeDet != NC_OK)
        {   // error queuing input tensor for inference
            printf("Could not queue detection inference\n");
            printf("Error from ncGraphQueueInferenceWithFifoElem is: %d\n", retCodeDet);
            exit(-1);
        }
        else
        {
            // the inference has been started, now read the output queue for the inference result
//            printf("---------------Successfully queued the detection inference for image-----------\n");


            unsigned int outFifoElemSize = 0;
            unsigned int optionSize = sizeof(outFifoElemSize);
            ncFifoGetOption(outFifoHandlePtr_det,  NC_RO_FIFO_ELEMENT_DATA_SIZE, &outFifoElemSize, &optionSize);

            float* resultDataFP32Ptr = (float*) malloc(outFifoElemSize);
            void* UserParamPtr = NULL;

            // read the output of the inference.  this will be in FP32 since that is how the fifos are created by default.
            retCodeDet = ncFifoReadElem(outFifoHandlePtr_det, (void*)resultDataFP32Ptr, &outFifoElemSize, &UserParamPtr);
            if (retCodeDet == NC_OK)
            {   // Successfully got the inference result.
                lpr_result_process(resultDataFP32Ptr, code_table);
                result_ = decodeLPRResults(code_table, mapping_table, 0.0);
            }
            delete imageBufFP32Ptr_det;
            free((void*)resultDataFP32Ptr);
        }

        return result_;

    }






}