#include "lpr_ncs.hpp"


int main(int argc, char *argv[]) {
    ros::init(argc, argv, "lpr_ncs2");
    ros::NodeHandle nh, priv_nh("~");

    lpr_ncs::LPR_NCS lpr_ncs2(priv_nh);
    ros::spin();
}