#!/usr/bin/env python

import math
import numpy as np

import rospy
from geometry_msgs.msg import Pose2D, Twist
from smach import State, StateMachine
from smach_ros import IntrospectionServer
from typing import List, Tuple


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
        State.__init__(self, outcomes=['ok', 'err'], input_keys=['targets'])
        self.pose_subscriber = rospy.Subscriber('pose2d', Pose2D, self.pose_callback, queue_size=10)
        self.bumper_subscriber = rospy.Subscriber('bumper', BumperEvent, self.bumper_callback, queue_size=10)
        self.twist_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.pose = None  # type: Pose2D
        self.speed = rospy.get_param('~move_speed')
        self.angular_speed = rospy.get_param('~turn_speed')
        self.position_threshold = rospy.get_param('~position_threshold')
        self.rate = rospy.Rate(10)

    def pose_callback(self, message):
        self.pose = message

    def execute(self, ud):
        targets = ud.targets  # type: List[Pose2D]

        while self.pose is None:
            rospy.logdebug('Waiting for odom...')
            rospy.sleep(1)

        for target in targets:
            self.drive_toward_target(target)
        self.drive_to_target(targets[-1])

        return 'ok'

    def drive_toward_target(self, target):  # type: (Pose2D) -> None
        """
        Drive toward the target. Return when the robot is *close to* the target (just before it would have to start
        slowing down)
        :param target:
        :return:
        """
        while True:
            dp, dtheta = self.get_deltas_to_target(target)
            if np.linalg.norm(dp) < self.position_threshold:
                return
            if np.linalg.norm(dp) < self.speed:
                return
            v = dp / np.linalg.norm(dp) * self.speed
            vtheta = dtheta if dtheta < self.angular_speed else np.sign(dtheta) * self.angular_speed
            self.publish_twist(v, vtheta)
            self.rate.sleep()

    def drive_to_target(self, target):  # type: (Pose2D) -> None
        """
        Drive entirely to the target pose
        :param target:
        :return:
        """
        while True:
            dp, dtheta = self.get_deltas_to_target(target)
            if np.linalg.norm(dp) < self.position_threshold:
                return
            if np.linalg.norm(dp) < self.speed:
                v = dp
            else:
                v = dp / np.linalg.norm(dp) * self.speed
            vtheta = dtheta if dtheta < self.angular_speed else np.sign(dtheta) * self.angular_speed
            self.publish_twist(v, vtheta)
            self.rate.sleep()

    def get_deltas_to_target(self, target):  # type: (Pose2D) -> Tuple[np.ndarray, float]
        pose = self.pose

        dx = target.x - pose.x
        dy = target.y - pose.y
        dtheta = math.atan2(dy, dx) - pose.theta
        if abs(dtheta) > np.pi:
            dtheta = dtheta - np.sign(dtheta) * 2 * np.pi
        return np.array([dx, dy]), dtheta

    def publish_twist(self, v, vtheta):  # type: (np.ndarray, float) -> None
        t = Twist()
        t.linear.x = max(0, v[0] * math.cos(self.pose.theta) + v[1] * math.sin(self.pose.theta))
        t.angular.z = vtheta
        self.twist_publisher.publish(t)


class MoveBackwardState(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err'])
        self.twist_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(10)
        self.duration = 2
        self.speed = rospy.get_param('~move_speed')

    def execute(self, ud):
        start_time = rospy.get_time()
        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < self.duration:
            t = Twist()
            t.linear.x = -self.speed
            self.twist_publisher.publish(t)
            self.rate.sleep()
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


sm = StateMachine(outcomes=['ok', 'err'], input_keys=['targets'])
with sm:
    StateMachine.add('MOVETO1', MoveState(), transitions={'ok': 'MOVETO1'}, remapping={'targets': 'targets'})

sis = IntrospectionServer('smach_server', sm, '/SM_ROOT')
sis.start()

# sm.execute({'theta1': 3.14159/2, 'theta2': 0.0})

initial_pose = get_odom()
pose1 = Pose2D(x=initial_pose.x, y=initial_pose.y)
pose2 = Pose2D(x=initial_pose.x - 2, y=initial_pose.y + 2)
pose3 = Pose2D(x=initial_pose.x - 2, y=initial_pose.y)

sm.execute({
    'targets': [
                   Pose2D(x=initial_pose.x, y=initial_pose.y),
                   Pose2D(x=initial_pose.x - 1, y=initial_pose.y + 1),
                   Pose2D(x=initial_pose.x - 2, y=initial_pose.y + 2),
                   Pose2D(x=initial_pose.x - 3, y=initial_pose.y + 3),
                   Pose2D(x=initial_pose.x - 4, y=initial_pose.y + 4),
                   Pose2D(x=initial_pose.x - 5, y=initial_pose.y + 5),
                   Pose2D(x=initial_pose.x - 2, y=initial_pose.y)
               ]*100,
})
