""" import rospy
from std_msgs.msg import Float64
import math
from FrankaTestBallPosition import Tutorial

class PublisherNode:
    joint = [
        '/franka/joint1_position_controller/command',
        '/franka/joint2_position_controller/command',
        '/franka/joint3_position_controller/command',
        '/franka/joint4_position_controller/command',
        '/franka/joint5_position_controller/command',
        '/franka/joint6_position_controller/command',
        '/franka/joint7_position_controller/command',
    ]

    def publish(self, action):
        for joint_position in action:
            pub = rospy.Publisher(joint[i], Float64, queue_size=10)
            pub.publish(joint_position)
        return object


def talker():
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

    movespeed = 0.05
    modnumber = math.pi

    position1 = 0
    position2 = modnumber/2
    position3 = modnumber/3
    position4 = modnumber/4
    position5 = 1
    position6 = 1.3
    position7 = 1.5
    position8 = 0
    position9 = 1

    t = Tutorial()



    while not rospy.is_shutdown():
        position1 += movespeed
        position1 %= modnumber

        position2 += movespeed
        position2 %= modnumber/2

        position3 += movespeed
        position3 %= modnumber-0.5

        position4 += movespeed
        position4 %= modnumber+2

        position5 += movespeed
        position5 %= modnumber/2

        position6 += movespeed
        position6 %= modnumber-0.5

        position7 += movespeed
        position7 %= modnumber+2

        position8 += 0.05
        position8 %= modnumber

        position9 += 0.01
        position9 %= 1

        # rospy.loginfo(position9)
        t.distance()

        # pub1.publish(position1)
        # pub2.publish(position2)
        # pub3.publish(position3)
        pub4.publish(-position4)
        # pub5.publish(position5)
        # pub6.publish(position6)
        # pub7.publish(position7)
        # pub8.publish(position8)

        pub9.publish(position9/10)

        rate.sleep()

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
 """