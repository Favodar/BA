import rospy
from std_msgs.msg import Float64
import math
from FrankaTestBallPosition import Tutorial
import random

def minimizer():
    joint1 = '/franka/joint1_position_controller/command'
    joint2 = '/franka/joint2_position_controller/command'
    joint3 = '/franka/joint3_position_controller/command'
    joint4 =  '/franka/joint4_position_controller/command'
    joint5 =  '/franka/joint5_position_controller/command'
    joint6 =  '/franka/joint6_position_controller/command'
    joint7 =  '/franka/joint7_position_controller/command'

    gripper = '/franka/gripper_position_controller/command'
    gripperwidth = '/franka/gripper_width'


    pub1 = rospy.Publisher(joint1, Float64, queue_size=10)
    pub2 = rospy.Publisher(joint2, Float64, queue_size=10)
    pub3 = rospy.Publisher(joint3, Float64, queue_size=10)
    pub4 = rospy.Publisher(joint4, Float64, queue_size=10)
    pub5 = rospy.Publisher(joint5, Float64, queue_size=10)
    pub6 = rospy.Publisher(joint6, Float64, queue_size=10)
    pub7 = rospy.Publisher(joint7, Float64, queue_size=10)
    pub8 = rospy.Publisher(gripper, Float64, queue_size=10)
    pub9 = rospy.Publisher(gripperwidth, Float64, queue_size=10)

    rospy.init_node('franka_publisher_node', anonymous=True)
    rate = rospy.Rate(100)

    movespeed = 0.00
    modnumber = math.pi

    position1 = 0.05
    position2 = 0
    position3 = modnumber/3
    position4 = modnumber/4
    position5 = 1
    position6 = 1.3
    position7 = 1.5
    position8 = 0
    position9 = 1

    t = Tutorial()


    try1 = position1
    try2 = position2
    try3 = position3
    try4 = position4

    tweak_amount = 0.1
    best_out = 1000
    best1 = position1
    best2 = position2
    best3 = position3
    best4 = position4


    check_rate = 1
    check_counter = 0
    counter_limit = 20 #rospy.Rate(1)
    failcounter = 1
    settle_counter = 0

    done = False

    while not rospy.is_shutdown():

        check_counter+=1
        settle_counter+=1

        if(settle_counter>200 and check_counter>=counter_limit):
            check_counter = 0
            out = t.distance()
            if(out<0.1):
                done = True
            #t.show_gazebo_models()
            if(out<best_out):
                failcounter = 1
                best_out = out
                best1 = try1
                best2 = try2
                best3 = try3
                best4 = try4
            elif not done:
                failcounter += 1


            try1 = best1 + tweak_amount*(random.random()-0.5)*failcounter
            try2 = best2 + tweak_amount*(random.random()-0.5)*failcounter
            try3 = best3 + tweak_amount*(random.random()-0.5)*failcounter
            try4 = best4 + tweak_amount*(random.random()-0.5)*failcounter

        pub1.publish(0)
        pub2.publish(try3)

        pub3.publish(0)
        pub4.publish(try1)

        pub5.publish(0)
        pub6.publish(try2)
        pub7.publish(try4)
        pub8.publish(0)
        pub9.publish(0)

        rate.sleep()

if __name__ == '__main__':
    try:
        minimizer()
    except rospy.ROSInterruptException:
        pass
