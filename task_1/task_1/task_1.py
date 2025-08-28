import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import math
import heapq
import numpy as np
import tf2_ros
from tf2_ros import TransformException
from tf_transformations import euler_from_quaternion


class AMRPlannerNode(Node):
    def __init__(self):
        super().__init__('task_1')

        # QoS for latched map data
        map_qos = QoSProfile(depth=1)
        map_qos.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

        # Publishers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.raw_path_pub = self.create_publisher(Path, '/path', 10)
        self.simplified_path_pub = self.create_publisher(Path, 'processed_path', 10)

        # Subscriptions
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, map_qos)
        self.goal_sub = self.create_subscription(PoseStamped, 'goal_pose', self.goal_callback, 10)
        self.laser_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # TF handling
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Periodic control loop
        self.timer = self.create_timer(0.1, self.control_loop)

        # Internal state
        self.map = None
        self.map_array = None
        self.inflation_radius = int(0.4417 / 0.05)

        self.path_waypoints = []
        self.current_index = 0
        self.have_path = False
        self.obstacle_points = []

        # Robot state
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Tuning params
        self.k_attract = 3.5
        self.k_repulse = 0.3
        self.waypoint_tolerance = 0.3
        self.goal_tolerance = 0.1
        self.yaw_tolerance = 0.1

        self.goal_yaw = None

        self.get_logger().info("Task 1 Planner Node is active.")

    # ---------------- Map Handling ---------------- #
    def map_callback(self, msg):
        """Receive map, inflate obstacles for safer planning."""
        self.map = msg
        self.map_array = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_array = self.inflate_obstacles(self.map_array)
        self.get_logger().info("Map received and processed with inflation.")

    def inflate_obstacles(self, grid):
        """Expand obstacles by inflation radius so paths keep clearance."""
        inflated = grid.copy()
        h, w = grid.shape
        radius = int(self.inflation_radius * 1.2)  # expand slightly more than nominal
        for y in range(h):
            for x in range(w):
                if grid[y, x] >= 50:
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                inflated[ny, nx] = 100
        return inflated

    # ---------------- Goal Handling ---------------- #
    def goal_callback(self, msg):
        """React to new goal by computing a path."""
        if self.map is None:
            self.get_logger().warn("No map available yet.")
            return

        # Find robot start pose from TF
        try:
            trans = self.tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
            start_x, start_y = trans.transform.translation.x, trans.transform.translation.y
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # Goal position and orientation
        goal_x, goal_y = msg.pose.position.x, msg.pose.position.y
        quat = msg.pose.orientation
        _, _, self.goal_yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        # Path planning
        path_cells = self.run_a_star(start_x, start_y, goal_x, goal_y)
        if path_cells:
            self.publish_raw_path(path_cells)

    # ---------------- Path Planning ---------------- #
    def run_a_star(self, sx, sy, gx, gy):
        """Run A* search from start to goal in map grid."""
        start = self.world_to_map(sx, sy)
        goal = self.world_to_map(gx, gy)

        if not self.valid_cell(start) or not self.valid_cell(goal):
            self.get_logger().warn("Invalid start or goal cell.")
            return None

        open_heap = []
        heapq.heappush(open_heap, (0, start))
        came_from = {}
        g_cost = {start: 0}
        f_cost = {start: self.heuristic(start, goal)}

        while open_heap:
            current = heapq.heappop(open_heap)[1]
            if current == goal:
                return self.reconstruct_path(came_from, current)

            for nb in self.neighbors(current):
                tentative = g_cost[current] + self.distance(current, nb)
                if nb not in g_cost or tentative < g_cost[nb]:
                    came_from[nb] = current
                    g_cost[nb] = tentative
                    f_cost[nb] = tentative + self.heuristic(nb, goal)
                    heapq.heappush(open_heap, (f_cost[nb], nb))
        return None

    def neighbors(self, cell):
        """Return free neighbor cells."""
        x, y = cell
        dirs = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (1,-1), (-1,1), (1,1)]
        return [
            (x+dx, y+dy)
            for dx, dy in dirs
            if self.in_bounds(x+dx, y+dy) and not self.is_blocked(x+dx, y+dy)
        ]

    def heuristic(self, a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def distance(self, a, b):
        return math.hypot(b[0]-a[0], b[1]-a[1])

    def reconstruct_path(self, came_from, node):
        """Rebuild full path from start to goal."""
        path = [node]
        while node in came_from:
            node = came_from[node]
            path.append(node)
        return list(reversed(path))

    # ---------------- Path Publishing ---------------- #
    def publish_raw_path(self, grid_cells):
        """Convert A* path cells to world coords and publish Path."""
        path_msg = Path()
        path_msg.header.frame_id = "map"
        path_msg.header.stamp = self.get_clock().now().to_msg()

        poses = []
        for mx, my in grid_cells:
            wx, wy = self.map_to_world(mx, my)
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.orientation.w = 1.0
            poses.append(pose)

        path_msg.poses = poses
        self.raw_path_pub.publish(path_msg)
        self.get_logger().info(f"Raw path published with {len(poses)} poses.")

        # Trigger post-processing
        self.refine_path(path_msg)

    def refine_path(self, path_msg):
        """Simplify raw path by removing redundant waypoints."""
        original = path_msg.poses
        if not original:
            self.get_logger().warn("Empty path received.")
            return

        filtered = [original[0]]
        for i in range(1, len(original)-1):
            prev, curr, nxt = original[i-1].pose.position, original[i].pose.position, original[i+1].pose.position
            a1 = math.atan2(curr.y - prev.y, curr.x - prev.x)
            a2 = math.atan2(nxt.y - curr.y, nxt.x - curr.x)
            if abs(a2 - a1) > 0.4:
                filtered.append(original[i])
        filtered.append(original[-1])

        new_path = Path()
        new_path.header = path_msg.header
        new_path.poses = filtered

        self.path_waypoints = [np.array([p.pose.position.x, p.pose.position.y]) for p in filtered]
        self.current_index = 0
        self.have_path = True

        self.simplified_path_pub.publish(new_path)
        self.get_logger().info(f"Processed path published with {len(filtered)} waypoints.")

    # ---------------- Motion Planning ---------------- #
    def scan_callback(self, msg):
        """Collect obstacle points from laser scan (robot frame)."""
        angle = msg.angle_min
        self.obstacle_points = []
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                self.obstacle_points.append(np.array([r * math.cos(angle), r * math.sin(angle)]))
            angle += msg.angle_increment

    def control_loop(self):
        """Periodic controller: follow path using attractive fields."""
        if not self.have_path:
            self.stop_robot()
            return

        try:
            trans: TransformStamped = self.tf_buffer.lookup_transform('odom', 'base_link', rclpy.time.Time())
            self.robot_x = trans.transform.translation.x
            self.robot_y = trans.transform.translation.y
            quat = trans.transform.rotation
            _, _, self.robot_yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
        except TransformException as e:
            self.get_logger().warn(f"TF error: {e}")
            self.stop_robot()
            return

        # Compute navigation control
        robot_pos = np.array([self.robot_x, self.robot_y])
        if not self.path_waypoints:
            self.stop_robot()
            return

        # Check final goal reached
        final_goal = self.path_waypoints[-1]
        if np.linalg.norm(robot_pos - final_goal) < self.goal_tolerance:
            yaw_error = math.atan2(math.sin(self.goal_yaw - self.robot_yaw), math.cos(self.goal_yaw - self.robot_yaw))
            if abs(yaw_error) < self.yaw_tolerance:
                self.get_logger().info("Goal reached with correct orientation.")
                self.stop_robot()
                return
            else:
                twist = Twist()
                twist.angular.z = max(-1.0, min(1.0, 2.5 * yaw_error))
                self.cmd_vel_pub.publish(twist)
                return

        # Otherwise, keep moving towards next waypoint
        target = self.path_waypoints[self.current_index]
        if np.linalg.norm(robot_pos - target) < self.waypoint_tolerance:
            self.current_index = min(self.current_index + 1, len(self.path_waypoints) - 1)
            target = self.path_waypoints[self.current_index]

        direction = target - robot_pos
        dist = np.linalg.norm(direction)
        v_attr = self.k_attract * (direction / dist) if dist > 0.01 else np.array([0.0, 0.0])

        desired_yaw = math.atan2(v_attr[1], v_attr[0])
        yaw_error = math.atan2(math.sin(desired_yaw - self.robot_yaw), math.cos(desired_yaw - self.robot_yaw))

        twist = Twist()
        if abs(yaw_error) > 0.3:
            twist.angular.z = max(-2.0, min(2.0, 3.0 * yaw_error))
            twist.linear.x = 0.0
        else:
            twist.linear.x = min(np.linalg.norm(v_attr), 0.7)
            twist.angular.z = max(-1.5, min(1.5, 2.0 * yaw_error))

        self.cmd_vel_pub.publish(twist)

    # ---------------- Helpers ---------------- #
    def stop_robot(self):
        """Publish zero velocity command."""
        self.cmd_vel_pub.publish(Twist())

    def world_to_map(self, x, y):
        origin = self.map.info.origin.position
        res = self.map.info.resolution
        return int((x - origin.x) / res), int((y - origin.y) / res)

    def map_to_world(self, mx, my):
        origin = self.map.info.origin.position
        res = self.map.info.resolution
        return mx * res + origin.x, my * res + origin.y

    def in_bounds(self, mx, my):
        return 0 <= mx < self.map.info.width and 0 <= my < self.map.info.height

    def is_blocked(self, mx, my):
        return self.map_array[my, mx] >= 50

    def valid_cell(self, cell):
        return self.in_bounds(*cell) and not self.is_blocked(*cell)


def main(args=None):
    rclpy.init(args=args)
    node = AMRPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

