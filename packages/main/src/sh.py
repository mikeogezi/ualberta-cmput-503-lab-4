#!/usr/bin/env python3

import os
import rospy
import roslib
from duckietown.dtros import DTROS, NodeType
from std_msgs.msg import String
from duckietown_msgs.msg import Twist2DStamped, LanePose, ButtonEvent, WheelsCmdStamped
import cv2
from sensor_msgs.msg import Image    
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rospkg
import time
import imutils

VERBOSE=False

y_low = np.array([20,43,46])
y_high = np.array([40,255,255])
r_low = np.array([155, 25, 0])
r_high = np.array([179, 255, 255])

class MySubscriberNode(DTROS):
    def __init__(self, node_name):
        super(MySubscriberNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        rospy.on_shutdown(self.on_shutdown)
        self.stre=False
        self.bridge=CvBridge()
        self.start = time.time()

        vehicle_name = os.environ['VEHICLE_NAME'] if 'VEHICLE_NAME' in os.environ else 'csc22917'
        self.pub_action = rospy.Publisher("/{}/joy_mapper_node/car_cmd".format(vehicle_name), Twist2DStamped, queue_size=10)
        self.img_sub = rospy.Subscriber("/{}/camera_node/image/compressed".format(vehicle_name), CompressedImage, self.callback)
        self.img_pub=rospy.Publisher('/{}/camera_node/filtered_image/compressed'.format(vehicle_name),CompressedImage,queue_size=1)
        self.img_pub2=rospy.Publisher('/{}/camera_node/filtered_image2/compressed'.format(vehicle_name),CompressedImage,queue_size=1)
        self.action_msg=Twist2DStamped()
        
        self.path = rospkg.RosPack().get_path("lab_4")
        self.stop_image=cv2.imread(self.path+ "/src/stop.png")
        #self.stop_image = cv2.cvtColor(self.stop_image, cv2.COLOR_BGR2GRAY)
        self.orb=cv2.ORB_create()
        self.kp2=self.orb.detect(self.stop_image,None)
        self.kp2,self.des2=self.orb.compute(self.stop_image,self.kp2)

    def on_shutdown(self):
        rospy.loginfo('Shutting down node...')
        self.stop()

    def stop(self):
        rospy.loginfo('Stopping...')
        self.pub_action.publish(Twist2DStamped())
        time.sleep(100)

    def can_detect_stop_sign_shape(self, filtered_image, ratio=1.):
        contours = cv2.findContours(filtered_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0 or M["m10"] == 0:
                continue

            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

            x, y, w, h = cv2.boundingRect(contour)
            ar = w / float(h)
            area = cv2.contourArea(contour)
            if area > 250 and len(approx) == 8:
                rospy.logwarn(area)
            if area < 1500 or area > 4000:
                continue
                
            if len(approx) == 8:
                rospy.logerr('{} red {}-gon detected'.format(area, len(approx)))
                return True
        return False

    def callback(self,ros_data):        
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, 1)
        h,w,d=image_np.shape
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        
        cut2 = image_np[int(h*3/4)-20:int(h), int(w/2)+50:int(w)-50]
        hsv2 = cv2.cvtColor(cut2, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv2,y_low,y_high)
        MY2 = cv2.moments(mask2,False)
        h2,w2,d2=cut2.shape
        screen_center2 =w2/2
        screen_hight2 = h2/2

        cut_top = image_np[0:int(h/2)-50, 0:int(w)]
        hsv_top = cv2.cvtColor(cut_top, cv2.COLOR_BGR2HSV)
        mask_top_r = cv2.inRange(hsv_top,r_low,r_high)
        d_shape = self.can_detect_stop_sign_shape(mask_top_r)
        
        cut_s = image_np[0:int(h/2), int(w/2)-100:int(w/2)+100]
        #cut_s = cv2.cvtColor(cut_s, cv2.COLOR_BGR2GRAY)
        
        '''
        kp1=self.orb.detect(cut_s,None)
        kp1,des1=self.orb.compute(cut_s,kp1)
    
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,self.des2, k=2)
        good = []
        for m,n in matches:
            if m.distance<0.75*n.distance:
                good.append([m])
        print(len(good))
        if len(good)>=100:
            self.action_msg.v=0
            self.action_msg.omega=0
            self.pub_action.publish(self.action_msg)
            print("Stop")  
            msg.data = np.array(cv2.imencode('.jpg', cut_s)[1]).tostring()

            time.sleep(100)     
        '''
        rospy.logwarn_throttle(0.5, time.time() - self.start)
        if d_shape and (time.time() - self.start) > 25.:
            self.stop()
        elif MY2['m00']>0:
            #self.action_msg.v=0.15
            cx2 = int(MY2['m10']/MY2['m00'])
            cy2 = int(MY2['m01']/MY2['m00'])
            cv2.circle(cut2,(cx2,cy2),20,(0,0,255),-1)
            cv2.circle(mask2,(cx2,cy2),20,(0,0,255),-1)
            
            error_x = cx2 - w2/2

            self.action_msg.v=0.10
            self.action_msg.omega = -error_x / 35
            
            msg.data = np.array(cv2.imencode('.jpg', cut2)[1]).tostring()
        else:
            self.action_msg.v=0.22
            self.action_msg.omega=0
            msg.data = np.array(cv2.imencode('.jpg', cut2)[1]).tostring()
        
        self.img_pub.publish(self.bridge.cv2_to_compressed_imgmsg(mask_top_r))    
        self.img_pub2.publish(self.bridge.cv2_to_compressed_imgmsg(cut_top))    
        self.pub_action.publish(self.action_msg)
    

        
        

if __name__ == '__main__':
    # create the node
    node = MySubscriberNode(node_name='my_subscriber_node')
    # keep spinning
    rospy.spin()

