#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose2D, Twist
from smach import State, StateMachine
from smach_ros import IntrospectionServer


class TurnState(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err'], input_keys=['target_theta'])
        self.pose_subscriber = rospy.Subscriber('pose2d', Pose2D, self.pose_callback, queue_size=10)
        self.twist_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.pose = None
        self.tolerance = rospy.get_param('~turn_tolerance')
        self.speed = rospy.get_param('~turn_speed')

    def pose_callback(self, message):
        self.pose = message

    def execute(self, userdata):
        while self.pose is None:
            rospy.logdebug('Waiting for odom...')
            rospy.sleep(1)
        rate = rospy.Rate(10)
        while True:
            diff = userdata.target_theta - self.pose.theta
            if abs(diff) < self.tolerance:
                break
            t = Twist()
            t.angular.z = diff/abs(diff)*self.speed
            self.twist_publisher.publish(t)
            rate.sleep()
        return 'ok'


rospy.init_node('control_node')

sm = StateMachine(outcomes=['ok', 'err'], input_keys=['theta1', 'theta2'])
with sm:
    StateMachine.add('TURN1', TurnState(), transitions={'ok': 'TURN2'}, remapping={'target_theta': 'theta1'})
    StateMachine.add('TURN2', TurnState(), transitions={'ok': 'TURN1'}, remapping={'target_theta': 'theta2'})

sis = IntrospectionServer('smach_server', sm, '/SM_ROOT')
sis.start()

sm.execute({'theta1': 3.14159/2, 'theta2': 0.0})
