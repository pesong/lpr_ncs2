//
// Created by pesong on 18-9-5.
//
#include "lpr_ncs2.hpp"

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "lpr_ncs2");
    ros::NodeHandle nh, priv_nh("~");

    lpr_ncs2::LPR_NCS lpr_ncs2(priv_nh);
    ros::spin();
}