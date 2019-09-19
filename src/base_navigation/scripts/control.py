#!/usr/bin/env python

import math
import numpy as np

import rospy
from geometry_msgs.msg import Pose2D, Twist
from kobuki_msgs.msg import BumperEvent, Sound
from sensor_msgs.msg import Joy
from smach import State, StateMachine
from smach_ros import IntrospectionServer
from typing import List, Tuple
from map import Map


sound_pub = publisher = rospy.Publisher('sound', Sound, queue_size=10)


def withsound(fn):

    def inner(*args, **kwargs):

        sound = Sound()
        sound.value = sound.ON
        sound_pub.publish(sound)
        fn(*args, **kwargs)

    return inner


class SubscriberValue:
    def __init__(self, topic, data_class, initial_value=None, transformation=None, queue_size=10):
        self._subscriber = rospy.Subscriber(topic, data_class, self._callback, queue_size=queue_size)
        self._initial_value = initial_value
        self._value = self._initial_value
        self._transformation = transformation
        self._unset = True

    def _callback(self, message):
        if self._transformation is None:
            self._value = message
        else:
            value, change = self._transformation(message)
            if change:
                self._unset = False
                self._value = value

    def reset(self):
        self._value = self._initial_value
        self._unset = True

    def wait(self, interval=1):
        while self._unset:
            rospy.logdebug('Waiting for message...')
            rospy.sleep(interval)

    @property
    def value(self):
        return self._value


class CollisionException(Exception):
    def __init__(self, bumper):  # type: (str) -> None
        self.bumper = bumper
        Exception.__init__(self, 'Bumper {} was pressed'.format(self.bumper))


class MoveState(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err', 'collision'], input_keys=['targets'], output_keys=['bumper_pressed'])
        self.pose = SubscriberValue('pose2d', Pose2D)
        self.bumper_pressed = SubscriberValue('bumper', BumperEvent, transformation=self.transform_bumper_message)
        self.twist_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.speed = rospy.get_param('~move_speed')
        self.angular_speed = rospy.get_param('~turn_speed')
        self.position_threshold = rospy.get_param('~position_threshold')
        self.rate = rospy.Rate(10)

    def transform_bumper_message(self, message):  # type: (BumperEvent) -> Tuple[str, bool]
        bumper_map = {
            message.LEFT: 'left',
            message.RIGHT: 'right',
            message.CENTER: 'center',
        }
        if message.state == message.PRESSED:
            return bumper_map[message.bumper], True
        else:
            return None, False

    @withsound
    def execute(self, ud):
        self.bumper_pressed.reset()
        targets = ud.targets  # type: List[Pose2D]
        ud.bumper_pressed = None  # type: str

        self.pose.wait()
        try:
            for target in targets:
                self.drive_toward_target(target)
            self.drive_to_target(targets[-1])
        except CollisionException as err:
            ud.bumper_pressed = err.bumper
            return 'collision'
        return 'ok'

    def drive_toward_target(self, target):  # type: (Pose2D) -> None
        """
        Drive toward the target. Return when the robot is *close to* the target (just before it would have to start
        slowing down)
        :param target:
        :return:
        """
        while True:
            if self.bumper_pressed is not None:
                raise CollisionException(self.bumper_pressed)
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
            if self.bumper_pressed is not None:
                raise CollisionException(self.bumper_pressed)
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
        pose = self.pose.value

        dx = target.x - pose.x
        dy = target.y - pose.y
        dtheta = math.atan2(dy, dx) - pose.theta
        if abs(dtheta) > np.pi:
            dtheta = dtheta - np.sign(dtheta) * 2 * np.pi
        return np.array([dx, dy]), dtheta

    def publish_twist(self, v, vtheta):  # type: (np.ndarray, float) -> None
        t = Twist()
        t.linear.x = max(0, v[0] * math.cos(self.pose.value.theta) + v[1] * math.sin(self.pose.value.theta))
        t.angular.z = vtheta
        self.twist_publisher.publish(t)


class MoveBackwardState(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err'])
        self.twist_publisher = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.rate = rospy.Rate(10)
        self.duration = 3
        self.speed = rospy.get_param('~move_speed')

    @withsound
    def execute(self, ud):
        start_time = rospy.get_time()
        while True:
            if rospy.is_shutdown():
                break
            if (rospy.get_time() - start_time) >= self.duration:
                break
            t = Twist()
            t.linear.x = -self.speed
            self.twist_publisher.publish(t)
            self.rate.sleep()
        self.twist_publisher.publish(Twist())
        return 'ok'


class PlayFinishSound(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err'])
        self.publisher = rospy.Publisher('sound', Sound, queue_size=10)

    def execute(self, userdata):
        sound = Sound()
        sound.value = sound.ON
        self.publisher.publish(sound)
        return 'ok'


class PlanTargets(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err'], input_keys=['world_map', 'goal'], output_keys=['targets', 'world_map'])
        self.pose = SubscriberValue('pose2d', Pose2D)

    @withsound
    def execute(self, ud):
        self.pose.wait()

        world_map = ud.world_map  # type: Map
        goal = ud.goal  # type: Pose2D
        targets = world_map.pathfind(
            start=np.array([self.pose.value.x, self.pose.value.y]),
            end=np.array([goal.x, goal.y]),
        )
        ud.targets = [
            Pose2D(x=target[0], y=target[1]) for target in targets
        ]
        return 'ok'


class UpdateMap(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err'], input_keys=['world_map'], output_keys=['world_map'])
        self.pose = SubscriberValue('pose2d', Pose2D)

    @withsound
    def execute(self, ud):
        self.pose.wait()
        world_map = ud.world_map  # type: Map
        world_map.fill_circle(np.array([self.pose.value.x, self.pose.value.y]), 1)
        world_map.image().show()
        return 'ok'


class WaitForJoystick(State):
    def __init__(self):
        State.__init__(self, outcomes=['ok', 'err'])
        self.joy_message = SubscriberValue('joy', Joy)

    @withsound
    def execute(self, ud):
        self.joy.wait()
        return 'ok'


rospy.init_node('control_node')

# sm = StateMachine(outcomes=['ok', 'err'], input_keys=['theta1', 'theta2'])
# with sm:
#     StateMachine.add('TURN1', TurnState(), transitions={'ok': 'TURN2'}, remapping={'target_theta': 'theta1'})
#     StateMachine.add('TURN2', TurnState(), transitions={'ok': 'TURN1'}, remapping={'target_theta': 'theta2'})


sm = StateMachine(outcomes=['ok', 'err'], input_keys=['world_map', 'goal'])
with sm:
    StateMachine.add('WAIT', WaitForJoystick(), transitions={'ok': 'PLAN'})
    StateMachine.add('PLAN', PlanTargets(), transitions={'ok': 'MOVE'})
    StateMachine.add('MOVE', MoveState(), transitions={'ok': 'PLAYSOUND', 'collision': 'UPDATEMAP'})
    StateMachine.add('PLAYSOUND', PlayFinishSound(), transitions={'ok': 'WAIT'})
    StateMachine.add('UPDATEMAP', UpdateMap(), transitions={'ok': 'BACKUP'})
    StateMachine.add('BACKUP', MoveBackwardState(), transitions={'ok': 'PLAN'})

sis = IntrospectionServer('smach_server', sm, '/SM_ROOT')
sis.start()

world_map = Map(np.array([0, 0]), scale=0.05, width=100, height=100)

sm.execute({
    'world_map': world_map,
    'goal': Pose2D(x=4, y=0),
})
