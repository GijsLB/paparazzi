#!/usr/bin/env python3
"""
Bebop Drone Obstacle Avoidance Integration

This script provides a bridge between the improved obstacle detection and
avoidance strategies and the Paparazzi UAV system. It can be used to test
and evaluate different approaches before implementing them in C for the
onboard system.

Features:
- Integration with Paparazzi through Ivy bus
- Real-time processing of video feed from Bebop
- Testing of avoidance strategies in simulation and real flights
- Data logging for performance analysis
"""

import sys
import os
import time
import numpy as np
import cv2
import argparse
import threading
import queue
import logging
from datetime import datetime

# Import our custom modules
# Make sure these are in the same directory or in the Python path
from improved_obstacle_detection import ImprovedObstacleDetector
from obstacle_avoidance_strategy import PotentialFieldStrategy, VectorFieldHistogramStrategy

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('bebop_obstacle_avoidance')

# Try to import Ivy-Python for communication with Paparazzi
try:
    from ivy.std_api import *
    HAS_IVY = True
except ImportError:
    logger.warning("Ivy-Python not found. Running in standalone mode only.")
    HAS_IVY = False

class BebopObstacleAvoidance:
    """Main class for Bebop obstacle avoidance integration"""
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            # General settings
            'mode': 'simulation',  # 'simulation', 'replay', 'live'
            'detection_method': 'color',  # 'color', 'advanced'
            'avoidance_strategy': 'vfh',  # 'reactive', 'potential', 'vfh'
            
            # Color filtering parameters (YUV color space)
            'y_min': 100, 'y_max': 200,
            'u_min': 120, 'u_max': 255,
            'v_min': 120, 'v_max': 255,
            
            # Detection parameters
            'detection_threshold': 0.18,
            'min_blob_size': 50,
            
            # Avoidance parameters
            'safety_distance': 1.0,  # meters
            'max_heading_change': np.radians(30),  # radians per update
            
            # Communication settings
            'ivy_bus': '127.255.255.255:2010',
            'ac_id': 1,  # Aircraft ID
            
            # Video settings
            'video_source': 0,  # Camera index or video file
            'video_width': 640,
            'video_height': 480,
            'fps': 30,
            
            # Logging settings
            'log_data': True,
            'log_dir': 'logs',
            'log_video': True,
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # Initialize components
        self._init_detector()
        self._init_strategy()
        
        # Initialize state
        self.drone_position = np.array([0.0, 0.0, 0.0])  # x, y, z in meters
        self.drone_heading = 0.0  # radians
        self.drone_speed = 0.0  # m/s
        
        # Initialize communication
        if HAS_IVY and (self.config['mode'] == 'live' or self.config['mode'] == 'simulation'):
            self._init_ivy()
        
        # Initialize video processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        self.video_thread = None
        self.processing_thread = None
        self.running = False
        
        # Initialize logging
        if self.config['log_data']:
            self._init_logging()
    
    def _init_detector(self):
        """Initialize the obstacle detector"""
        detector_config = {
            'y_min': self.config['y_min'],
            'y_max': self.config['y_max'],
            'u_min': self.config['u_min'],
            'u_max': self.config['u_max'],
            'v_min': self.config['v_min'],
            'v_max': self.config['v_max'],
            'min_blob_size': self.config['min_blob_size'],
            'detection_threshold': self.config['detection_threshold'],
        }
        
        if self.config['detection_method'] == 'advanced':
            # Use the advanced detector with additional features
            detector_config.update({
                'use_edge_detection': True,
                'use_optical_flow': True,
                'distance_estimation': True,
            })
        
        self.detector = ImprovedObstacleDetector(detector_config)
    
    def _init_strategy(self):
        """Initialize the avoidance strategy"""
        strategy_config = {
            'safety_distance': self.config['safety_distance'],
            'max_heading_change': self.config['max_heading_change'],
        }
        
        if self.config['avoidance_strategy'] == 'potential':
            self.strategy = PotentialFieldStrategy(strategy_config)
        elif self.config['avoidance_strategy'] == 'vfh':
            self.strategy = VectorFieldHistogramStrategy(strategy_config)
        else:  # reactive (similar to original orange_avoider)
            self.strategy = None  # Will use simple reactive avoidance
    
    def _init_ivy(self):
        """Initialize Ivy communication with Paparazzi"""
        if not HAS_IVY:
            return
            
        # Initialize Ivy
        IvyInit("BebopObstacleAvoidance", 
                "Bebop Obstacle Avoidance Ready", 
                0, lambda x, y: None, lambda x, y: None)
        
        # Start Ivy
        IvyStart(self.config['ivy_bus'])
        
        # Bind to messages
        IvyBindMsg(self._on_position_message, 
                  f"^({self.config['ac_id']}) ROTORCRAFT_FP .*")
        IvyBindMsg(self._on_status_message, 
                  f"^({self.config['ac_id']}) ROTORCRAFT_STATUS .*")
        
        logger.info(f"Connected to Ivy bus: {self.config['ivy_bus']}")
    
    def _init_logging(self):
        """Initialize data logging"""
        # Create log directory if it doesn't exist
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Create timestamp for log files
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create log file
        log_file = os.path.join(self.config['log_dir'], f"avoidance_log_{timestamp}.csv")
        self.log_file = open(log_file, 'w')
        
        # Write header
        header = "timestamp,x,y,z,heading,speed,obstacle_detected,obstacle_ratio,confidence\n"
        self.log_file.write(header)
        self.log_file.flush()
        
        # Initialize video writer if needed
        self.video_writer = None
        if self.config['log_video']:
            video_file = os.path.join(self.config['log_dir'], f"avoidance_video_{timestamp}.avi")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(
                video_file, fourcc, self.config['fps'],
                (self.config['video_width'], self.config['video_height'])
            )
    
    def _on_position_message(self, agent, *args):
        """Handle position messages from Paparazzi"""
        # Parse position message
        # Format depends on Paparazzi version, adjust as needed
        try:
            # Extract position (in mm) and convert to meters
            x = float(args[1]) / 1000.0
            y = float(args[2]) / 1000.0
            z = float(args[3]) / 1000.0
            
            # Extract heading (in radians)
            heading = float(args[7]) / 1000.0
            
            # Update drone state
            self.drone_position = np.array([x, y, z])
            self.drone_heading = heading
            
            logger.debug(f"Position update: ({x:.2f}, {y:.2f}, {z:.2f}), heading: {np.degrees(heading):.1f}°")
        except Exception as e:
            logger.error(f"Error parsing position message: {e}")
    
    def _on_status_message(self, agent, *args):
        """Handle status messages from Paparazzi"""
        # Parse status message
        # Format depends on Paparazzi version, adjust as needed
        try:
            # Extract relevant information
            # ...
            
            logger.debug("Status update received")
        except Exception as e:
            logger.error(f"Error parsing status message: {e}")
    
    def _send_nav_command(self, heading, speed):
        """Send navigation command to Paparazzi"""
        if not HAS_IVY:
            return
            
        try:
            # Convert heading to Paparazzi format (milliradians)
            heading_pprz = int(heading * 1000)
            
            # Send command
            # Format depends on Paparazzi version, adjust as needed
            IvySendMsg(f"dl NAVIGATION_REF {self.config['ac_id']} {heading_pprz} {int(speed*100)}")
            
            logger.debug(f"Sent navigation command: heading={np.degrees(heading):.1f}°, speed={speed:.2f}m/s")
        except Exception as e:
            logger.error(f"Error sending navigation command: {e}")
    
    def _video_capture_thread(self):
        """Thread for capturing video frames"""
        # Open video source
        if isinstance(self.config['video_source'], str):
            # Assume it's a video file
            cap = cv2.VideoCapture(self.config['video_source'])
        else:
            # Assume it's a camera index
            cap = cv2.VideoCapture(self.config['video_source'])
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['video_width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['video_height'])
        
        if not cap.isOpened():
            logger.error("Failed to open video source")
            return
        
        logger.info("Video capture started")
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                if isinstance(self.config['video_source'], str):
                    # End of video file
                    logger.info("End of video file reached")
                    break
                else:
                    # Camera error
                    logger.warning("Failed to read frame from camera")
                    time.sleep(0.1)
                    continue
            
            # Put frame in queue, skip if queue is full
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass
        
        cap.release()
        logger.info("Video capture stopped")
    
    def _processing_thread(self):
        """Thread for processing video frames"""
        logger.info("Processing thread started")
        
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                result = self.detector.process_frame(frame)
                
                # Create visualization
                vis_frame = self.detector.visualize_detection(frame, result)
                
                # Put result in queue
                self.result_queue.put((result, vis_frame), block=False)
                
                # Log data if enabled
                if self.config['log_data']:
                    self._log_data(result)
                
                # Log video if enabled
                if self.config['log_video'] and self.video_writer is not None:
                    self.video_writer.write(vis_frame)
                
                # Compute avoidance if strategy is available
                if self.strategy is not None and self.config['mode'] != 'replay':
                    # Convert detection result to obstacles for strategy
                    obstacles = self._detection_to_obstacles(result)
                    
                    # Compute avoidance vector
                    # For simplicity, we use a dummy goal ahead of the drone
                    goal = self.drone_position[:2] + np.array([
                        10.0 * np.cos(self.drone_heading),
                        10.0 * np.sin(self.drone_heading)
                    ])
                    
                    new_heading, speed = self.strategy.compute_avoidance_vector(
                        self.drone_position[:2], self.drone_heading,
                        obstacles, goal
                    )
                    
                    # Send command if in live mode
                    if self.config['mode'] == 'live':
                        self._send_nav_command(new_heading, speed)
                    
                    # Update drone state for simulation
                    if self.config['mode'] == 'simulation':
                        # Simple simulation update
                        self.drone_heading = new_heading
                        self.drone_speed = speed
                        
                        # Update position
                        movement = speed * 0.1 * np.array([
                            np.cos(self.drone_heading),
                            np.sin(self.drone_heading),
                            0.0  # Assume constant altitude
                        ])
                        self.drone_position += movement
                
                # Mark as done
                self.frame_queue.task_done()
                
            except queue.Empty:
                # No frames available
                pass
            except Exception as e:
                logger.error(f"Error in processing thread: {e}")
        
        logger.info("Processing thread stopped")
    
    def _detection_to_obstacles(self, result):
        """Convert detection result to obstacles for avoidance strategy"""
        obstacles = []
        
        # If we have distance estimates, use them
        if 'distances' in result and result['distances']:
            for dist_info in result['distances']:
                # Get position in drone's frame
                x, y = dist_info['position']
                distance = dist_info['distance']
                
                # Convert to world coordinates
                angle = np.arctan2(y - self.config['video_height']/2, 
                                  x - self.config['video_width']/2)
                
                obs_x = self.drone_position[0] + distance * np.cos(self.drone_heading + angle)
                obs_y = self.drone_position[1] + distance * np.sin(self.drone_heading + angle)
                
                # Add obstacle with estimated radius
                obstacles.append((obs_x, obs_y, 0.3))  # Assume 30cm radius
        
        # If no distance estimates but obstacle detected, create a virtual obstacle
        elif result['obstacle_detected']:
            # Create obstacle in front of drone
            distance = 2.0  # Assume 2m distance
            obs_x = self.drone_position[0] + distance * np.cos(self.drone_heading)
            obs_y = self.drone_position[1] + distance * np.sin(self.drone_heading)
            
            # Add obstacle
            obstacles.append((obs_x, obs_y, 0.5))  # Assume 50cm radius
        
        return obstacles
    
    def _log_data(self, result):
        """Log detection and avoidance data"""
        if not self.config['log_data'] or self.log_file is None:
            return
            
        # Get current timestamp
        timestamp = time.time()
        
        # Format data
        data = f"{timestamp},{self.drone_position[0]},{self.drone_position[1]},{self.drone_position[2]},"
        data += f"{self.drone_heading},{self.drone_speed},{int(result['obstacle_detected'])},"
        data += f"{result['obstacle_ratio']},{result['confidence']}\n"
        
        # Write to log file
        self.log_file.write(data)
        self.log_file.flush()
    
    def start(self):
        """Start the obstacle avoidance system"""
        if self.running:
            logger.warning("System is already running")
            return
            
        self.running = True
        
        # Start video capture thread
        self.video_thread = threading.Thread(target=self._video_capture_thread)
        self.video_thread.daemon = True
        self.video_thread.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_thread)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Obstacle avoidance system started")
    
    def stop(self):
        """Stop the obstacle avoidance system"""
        if not self.running:
            logger.warning("System is not running")
            return
            
        self.running = False
        
        # Wait for threads to finish
        if self.video_thread is not None:
            self.video_thread.join(timeout=2.0)
        
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=2.0)
        
        # Close log file
        if self.config['log_data'] and self.log_file is not None:
            self.log_file.close()
            self.log_file = None
        
        # Close video writer
        if self.config['log_video'] and self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        
        # Disconnect from Ivy
        if HAS_IVY:
            IvyStop()
        
        logger.info("Obstacle avoidance system stopped")
    
    def run_interactive(self):
        """Run the system with interactive visualization"""
        self.start()
        
        try:
            while self.running:
                try:
                    # Get processed result
                    result, vis_frame = self.result_queue.get(timeout=0.1)
                    
                    # Display frame
                    cv2.imshow("Obstacle Avoidance", vis_frame)
                    
                    # Process keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    
                    # Mark as done
                    self.result_queue.task_done()
                    
                except queue.Empty:
                    # No results available
                    pass
                
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        finally:
            self.stop()
            cv2.destroyAllWindows()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Bebop Obstacle Avoidance")
    
    # Mode selection
    parser.add_argument("--mode", choices=["simulation", "replay", "live"], default="simulation",
                       help="Operating mode")
    
    # Detection method
    parser.add_argument("--detection", choices=["color", "advanced"], default="color",
                       help="Detection method")
    
    # Avoidance strategy
    parser.add_argument("--strategy", choices=["reactive", "potential", "vfh"], default="vfh",
                       help="Avoidance strategy")
    
    # Video source
    parser.add_argument("--video", default=0,
                       help="Video source (camera index or file path)")
    
    # Color thresholds
    parser.add_argument("--y_min", type=int, default=100, help="Y min threshold")
    parser.add_argument("--y_max", type=int, default=200, help="Y max threshold")
    parser.add_argument("--u_min", type=int, default=120, help="U min threshold")
    parser.add_argument("--u_max", type=int, default=255, help="U max threshold")
    parser.add_argument("--v_min", type=int, default=120, help="V min threshold")
    parser.add_argument("--v_max", type=int, default=255, help="V max threshold")
    
    # Detection parameters
    parser.add_argument("--threshold", type=float, default=0.18,
                       help="Detection threshold")
    
    # Logging
    parser.add_argument("--no-log", action="store_true",
                       help="Disable data logging")
    parser.add_argument("--no-video-log", action="store_true",
                       help="Disable video logging")
    
    args = parser.parse_args()
    
    # Prepare configuration
    config = {
        'mode': args.mode,
        'detection_method': args.detection,
        'avoidance_strategy': args.strategy,
        'y_min': args.y_min,
        'y_max': args.y_max,
        'u_min': args.u_min,
        'u_max': args.u_max,
        'v_min': args.v_min,
        'v_max': args.v_max,
        'detection_threshold': args.threshold,
        'log_data': not args.no_log,
        'log_video': not args.no_video_log,
    }
    
    # Set video source
    if args.video.isdigit():
        config['video_source'] = int(args.video)
    else:
        config['video_source'] = args.video
    
    # Create and run system
    system = BebopObstacleAvoidance(config)
    system.run_interactive()

if __name__ == "__main__":
    main()
