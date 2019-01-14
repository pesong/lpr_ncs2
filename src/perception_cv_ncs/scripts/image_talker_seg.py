#!/usr/bin/env python
# license removed for brevity
import os
import rospy
import skimage.transform
import skimage.io
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

IMAGE_PATH_ROOT = '/dl/model/seg/caffe/ncs_fcns/demo_test/CS/'


def talker():
    pub = rospy.Publisher('/rgb_image_hd', Image, queue_size=1)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(3)
    bridge = CvBridge()
    while not rospy.is_shutdown():
        for IMAGE_PATH in os.listdir(IMAGE_PATH_ROOT):
            img = skimage.io.imread(os.path.join(IMAGE_PATH_ROOT + IMAGE_PATH))
            # img = img[:, :, ::-1]
            # cv2.imshow("talker", img)
            cv2.waitKey(1)
            pub.publish(bridge.cv2_to_imgmsg(img, "bgr8"))
            rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
