#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import PoseArray, Pose, PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from geometry_msgs.msg import TransformStamped
import tf_transformations
import math
import random

class ParticleFilter(Node):
    def __init__(self):
        super().__init__('particle_filter')
        
        # Parameters
        self.num_particles = 500
        self.map_received = False
        
        # Initialize particles
        self.particles = np.zeros((self.num_particles, 3))  # [x, y, theta]
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        # ROS2 subscriptions
        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.map_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        self.initial_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/initialpose', self.initial_pose_callback, 10)
        
        # Publishers
        self.particle_pub = self.create_publisher(PoseArray, '/particlecloud', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/amcl_pose', 10)
        
        # TF
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Timer for main loop
        self.create_timer(0.1, self.update_filter)  # 10 Hz
        
        # State variables
        self.last_odom = None
        self.occupancy_map = None
        self.map_info = None
        
        self.get_logger().info("Particle Filter initialized with {} particles".format(self.num_particles))

    def map_callback(self, msg):
        """Store the occupancy grid map"""
        self.occupancy_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info
        self.map_received = True
        self.get_logger().info("Map received: {}x{}".format(msg.info.width, msg.info.height))

    def initial_pose_callback(self, msg):
        """Initialize particles around the given pose"""
        if not self.map_received:
            self.get_logger().warn("Map not received yet, cannot initialize particles")
            return
            
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        quaternion = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([
            quaternion.x, quaternion.y, quaternion.z, quaternion.w])
        
        # Initialize particles with some spread around the initial pose
        self.particles[:, 0] = np.random.normal(x, 0.5, self.num_particles)  # x with 0.5m std
        self.particles[:, 1] = np.random.normal(y, 0.5, self.num_particles)  # y with 0.5m std
        self.particles[:, 2] = np.random.normal(yaw, 0.2, self.num_particles)  # theta with 0.2 rad std
        
        # Reset weights
        self.weights = np.ones(self.num_particles) / self.num_particles
        
        self.get_logger().info(f"Particles initialized around pose: ({x:.2f}, {y:.2f}, {yaw:.2f})")

    def odom_callback(self, msg):
        """Store odometry for motion model"""
        current_odom = msg
        
        if self.last_odom is not None:
            # Calculate odometry delta
            delta_x = current_odom.pose.pose.position.x - self.last_odom.pose.pose.position.x
            delta_y = current_odom.pose.pose.position.y - self.last_odom.pose.pose.position.y
            
            # Get current and previous orientations
            curr_quat = current_odom.pose.pose.orientation
            prev_quat = self.last_odom.pose.pose.orientation
            
            _, _, curr_yaw = tf_transformations.euler_from_quaternion([
                curr_quat.x, curr_quat.y, curr_quat.z, curr_quat.w])
            _, _, prev_yaw = tf_transformations.euler_from_quaternion([
                prev_quat.x, prev_quat.y, prev_quat.z, prev_quat.w])
            
            delta_theta = curr_yaw - prev_yaw
            
            # Normalize angle difference
            delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
            
            # Apply motion model if there's significant movement
            if abs(delta_x) > 0.01 or abs(delta_y) > 0.01 or abs(delta_theta) > 0.01:
                self.motion_model(delta_x, delta_y, delta_theta)
        
        self.last_odom = current_odom

    def laser_callback(self, msg):
        """Store laser scan for sensor model"""
        self.last_laser = msg

    def motion_model(self, delta_x, delta_y, delta_theta):
        """Apply motion model to all particles"""
        if not hasattr(self, 'particles') or len(self.particles) == 0:
            return
            
        # Add noise to motion (you can tune these values)
        noise_x = np.random.normal(0, 0.1, self.num_particles)
        noise_y = np.random.normal(0, 0.1, self.num_particles)
        noise_theta = np.random.normal(0, 0.05, self.num_particles)
        
        # Update particle positions
        self.particles[:, 0] += delta_x + noise_x
        self.particles[:, 1] += delta_y + noise_y
        self.particles[:, 2] += delta_theta + noise_theta
        
        # Normalize angles
        self.particles[:, 2] = np.arctan2(np.sin(self.particles[:, 2]), 
                                         np.cos(self.particles[:, 2]))

    def sensor_model(self, laser_scan):
        """Update particle weights based on laser scan"""
        if not self.map_received or self.occupancy_map is None:
            return
            
        # This is a simplified sensor model - you'll want to improve this
        for i in range(self.num_particles):
            # For now, just give equal weights (placeholder)
            # In a real implementation, you'd:
            # 1. Ray-cast from particle pose
            # 2. Compare expected vs actual laser readings
            # 3. Update weight based on similarity
            self.weights[i] = 1.0
        
        # Normalize weights
        self.weights /= np.sum(self.weights)

    def resample_particles(self):
        """Resample particles based on weights"""
        # Low variance resampling
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_pose(self):
        """Estimate robot pose from particles"""
        # Weighted average of particles
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        
        # For angle, use circular mean
        cos_theta = np.average(np.cos(self.particles[:, 2]), weights=self.weights)
        sin_theta = np.average(np.sin(self.particles[:, 2]), weights=self.weights)
        theta = np.arctan2(sin_theta, cos_theta)
        
        return x, y, theta

    def publish_particles(self):
        """Publish particle cloud for visualization"""
        if not hasattr(self, 'particles'):
            return
            
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'map'
        
        for i in range(self.num_particles):
            pose = Pose()
            pose.position.x = self.particles[i, 0]
            pose.position.y = self.particles[i, 1]
            pose.position.z = 0.0
            
            # Convert angle to quaternion
            quat = tf_transformations.quaternion_from_euler(0, 0, self.particles[i, 2])
            pose.orientation.x = quat[0]
            pose.orientation.y = quat[1]
            pose.orientation.z = quat[2]
            pose.orientation.w = quat[3]
            
            pose_array.poses.append(pose)
        
        self.particle_pub.publish(pose_array)

    def publish_pose(self, x, y, theta):
        """Publish estimated pose"""
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        
        pose_msg.pose.pose.position.x = x
        pose_msg.pose.pose.position.y = y
        pose_msg.pose.pose.position.z = 0.0
        
        quat = tf_transformations.quaternion_from_euler(0, 0, theta)
        pose_msg.pose.pose.orientation.x = quat[0]
        pose_msg.pose.pose.orientation.y = quat[1]
        pose_msg.pose.pose.orientation.z = quat[2]
        pose_msg.pose.pose.orientation.w = quat[3]
        
        self.pose_pub.publish(pose_msg)

    def update_filter(self):
        """Main filter update loop"""
        if not self.map_received:
            return
            
        # Motion update (if we have odometry)
        if hasattr(self, 'last_odom') and self.last_odom is not None:
            # For now, we'll skip motion update - you'll implement this next
            pass
        
        # Sensor update (if we have laser scan)
        if hasattr(self, 'last_laser'):
            self.sensor_model(self.last_laser)
        
        # Resample if needed (check effective sample size)
        effective_sample_size = 1.0 / np.sum(self.weights ** 2)
        if effective_sample_size < self.num_particles / 2:
            self.resample_particles()
        
        # Publish results
        self.publish_particles()
        
        # Estimate and publish pose
        if hasattr(self, 'particles') and len(self.particles) > 0:
            x, y, theta = self.estimate_pose()
            self.publish_pose(x, y, theta)

def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilter()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()