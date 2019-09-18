#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Pose2D, Twist
from smach import State, StateMachine
from smach_ros import IntrospectionServer
import tf

import math
import numpy as np


def get_odom():  # type: () -> Pose2D
    get_odom.pose = None

    def pose_callback(pose):
        get_odom.pose = pose

    _ = rospy.Subscriber('pose2d', Pose2D, pose_callback, queue_size=10)
    while get_odom.pose is None:
        rospy.sleep(0.5)

    return get_odom.pose


class MoveState(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err'], input_keys=['target'])
        self.pose_subscriber = rospy.Subscriber('pose2d', Pose2D, self.pose_callback, queue_size=10)
        self.twist_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.pose = None  # type: Pose2D
        self.speed = rospy.get_param('~move_speed')
        self.angular_speed = rospy.get_param('~turn_speed')
        self.position_threshold = rospy.get_param('~position_threshold')

    def pose_callback(self, message):
        self.pose = message

    def execute(self, ud):
        target = ud.target  # type: Pose2D

        while self.pose is None:
            rospy.logdebug('Waiting for odom...')
            rospy.sleep(1)

        rate = rospy.Rate(10)
        while True:
            pose = self.pose

            dx = target.x - pose.x
            dy = target.y - pose.y
            dtheta = math.atan2(dy, dx) - pose.theta
            if abs(dtheta) > np.pi:
                dtheta = dtheta - np.sign(dtheta)*2*np.pi

            if dx**2 + dy**2 < self.position_threshold**2:
                break

            v = np.array([dx, dy])
            if np.linalg.norm(v) > self.speed:
                v = v / np.linalg.norm(v) * self.speed
            vtheta = dtheta if dtheta < self.angular_speed else np.sign(dtheta) * self.angular_speed

            t = Twist()
            t.linear.x = max(0, v[0]*math.cos(pose.theta) + v[1]*math.sin(pose.theta))
            t.angular.z = vtheta
            self.twist_publisher.publish(t)
            rate.sleep()

        return 'ok'


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

# sm = StateMachine(outcomes=['ok', 'err'], input_keys=['theta1', 'theta2'])
# with sm:
#     StateMachine.add('TURN1', TurnState(), transitions={'ok': 'TURN2'}, remapping={'target_theta': 'theta1'})
#     StateMachine.add('TURN2', TurnState(), transitions={'ok': 'TURN1'}, remapping={'target_theta': 'theta2'})

sm = StateMachine(outcomes=['ok', 'err'], input_keys=['target1', 'target2', 'target3'])
with sm:
    StateMachine.add('MOVETO1', MoveState(), transitions={'ok': 'MOVETO2'}, remapping={'target': 'target1'})
    StateMachine.add('MOVETO2', MoveState(), transitions={'ok': 'MOVETO3'}, remapping={'target': 'target2'})
    StateMachine.add('MOVETO3', MoveState(), transitions={'ok': 'MOVETO1'}, remapping={'target': 'target3'})

sis = IntrospectionServer('smach_server', sm, '/SM_ROOT')
sis.start()

# sm.execute({'theta1': 3.14159/2, 'theta2': 0.0})

initial_pose = get_odom()
pose1 = Pose2D(x=initial_pose.x, y=initial_pose.y)
pose2 = Pose2D(x=initial_pose.x - 2, y=initial_pose.y + 2)
pose3 = Pose2D(x=initial_pose.x - 2, y=initial_pose.y)

sm.execute({'target1': pose2, 'target2': pose3, 'target3': pose1})
