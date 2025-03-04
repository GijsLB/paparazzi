#!/usr/bin/env python3
"""
Improved Obstacle Detection for Paparazzi UAV Orange Avoider

This script provides enhanced obstacle detection methods that can be integrated
with the Paparazzi UAV system to improve the orange avoider module.

Features:
- Multiple detection methods beyond simple color filtering
- Adaptive thresholding based on lighting conditions
- Obstacle classification by type and threat level
- Distance estimation to obstacles
"""

import cv2
import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from datetime import datetime

class ImprovedObstacleDetector:
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            # Color filtering parameters (YUV color space)
            'y_min': 100, 'y_max': 200,
            'u_min': 120, 'u_max': 255,
            'v_min': 120, 'v_max': 255,
            
            # Detection parameters
            'min_blob_size': 50,
            'detection_threshold': 0.18,
            'confidence_max': 5,
            
            # Adaptive parameters
            'enable_adaptive_threshold': True,
            'history_length': 10,
            
            # Advanced features
            'use_edge_detection': True,
            'use_optical_flow': False,
            'distance_estimation': True
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
            
        # Initialize state
        self.frame_history = []
        self.detection_history = []
        self.obstacle_confidence = 0
        
    def color_filter(self, image):
        """Apply color filtering in YUV color space"""
        # Convert to YUV if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        else:
            # Handle YUV422 format from Paparazzi
            # This is a simplified conversion - actual implementation depends on exact format
            yuv = image
            
        # Create mask based on color thresholds
        mask = cv2.inRange(
            yuv,
            (self.config['y_min'], self.config['u_min'], self.config['v_min']),
            (self.config['y_max'], self.config['u_max'], self.config['v_max'])
        )
        
        return mask
    
    def adaptive_threshold_update(self, image):
        """Dynamically adjust thresholds based on lighting conditions"""
        if not self.config['enable_adaptive_threshold']:
            return
            
        # Add current frame to history
        if len(image.shape) == 3 and image.shape[2] == 3:
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            self.frame_history.append(yuv)
        else:
            self.frame_history.append(image)
            
        # Keep history at fixed length
        if len(self.frame_history) > self.config['history_length']:
            self.frame_history.pop(0)
            
        # Skip if not enough history
        if len(self.frame_history) < 3:
            return
            
        # Calculate average luminance (Y channel)
        avg_y = np.mean([np.mean(frame[:,:,0]) for frame in self.frame_history])
        
        # Adjust thresholds based on average luminance
        # Brighter environment -> tighter thresholds
        # Darker environment -> wider thresholds
        if avg_y > 150:  # Bright environment
            self.config['y_min'] = max(80, min(120, avg_y - 50))
            self.config['y_max'] = min(255, avg_y + 50)
        else:  # Darker environment
            self.config['y_min'] = max(50, min(100, avg_y - 70))
            self.config['y_max'] = min(220, avg_y + 70)
    
    def edge_based_detection(self, image):
        """Use edge detection to find obstacle boundaries"""
        if not self.config['use_edge_detection']:
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to connect nearby lines
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        return dilated
    
    def find_contours(self, mask):
        """Find contours in the mask and filter by size"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter by size
        filtered_contours = []
        for contour in contours:
            if cv2.contourArea(contour) > self.config['min_blob_size']:
                filtered_contours.append(contour)
                
        return filtered_contours
    
    def estimate_distances(self, contours, image_width, focal_length=500, known_width=0.1):
        """Estimate distance to obstacles based on apparent size"""
        if not self.config['distance_estimation']:
            return []
            
        distances = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Estimate distance using simple pinhole camera model
            # distance = (known_width * focal_length) / apparent_width
            distance = (known_width * focal_length) / w
            
            distances.append({
                'contour': contour,
                'distance': distance,
                'position': (x + w/2, y + h/2)  # center point
            })
            
        return distances
    
    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        # Update adaptive thresholds
        self.adaptive_threshold_update(frame)
        
        # Apply color filtering
        mask = self.color_filter(frame)
        
        # Get edge information if enabled
        edges = self.edge_based_detection(frame)
        if edges is not None:
            # Combine color and edge information
            combined_mask = cv2.bitwise_or(mask, edges)
        else:
            combined_mask = mask
        
        # Find contours
        contours = self.find_contours(combined_mask)
        
        # Calculate obstacle ratio (similar to orange_avoider.c)
        pixel_count = np.sum(combined_mask > 0)
        total_pixels = frame.shape[0] * frame.shape[1]
        obstacle_ratio = pixel_count / total_pixels
        
        # Update confidence
        if obstacle_ratio > self.config['detection_threshold']:
            self.obstacle_confidence = max(0, self.obstacle_confidence - 2)
        else:
            self.obstacle_confidence = min(self.config['confidence_max'], self.obstacle_confidence + 1)
        
        # Estimate distances if enabled
        distances = []
        if contours and self.config['distance_estimation']:
            distances = self.estimate_distances(contours, frame.shape[1])
        
        # Prepare result
        result = {
            'obstacle_detected': obstacle_ratio > self.config['detection_threshold'],
            'obstacle_ratio': obstacle_ratio,
            'confidence': self.obstacle_confidence,
            'contours': contours,
            'distances': distances,
            'mask': combined_mask
        }
        
        return result
    
    def visualize_detection(self, frame, result):
        """Create a visualization of the detection results"""
        # Create a copy for drawing
        vis_frame = frame.copy()
        
        # Draw mask as overlay
        mask_overlay = cv2.cvtColor(result['mask'], cv2.COLOR_GRAY2BGR)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, mask_overlay, 0.3, 0)
        
        # Draw contours
        cv2.drawContours(vis_frame, result['contours'], -1, (0, 255, 0), 2)
        
        # Draw distance information
        for dist_info in result['distances']:
            contour = dist_info['contour']
            distance = dist_info['distance']
            position = dist_info['position']
            
            # Draw center point
            cv2.circle(vis_frame, (int(position[0]), int(position[1])), 5, (255, 0, 0), -1)
            
            # Draw distance text
            cv2.putText(vis_frame, f"{distance:.2f}m", 
                       (int(position[0]), int(position[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add overall status
        status_text = "OBSTACLE" if result['obstacle_detected'] else "CLEAR"
        conf_text = f"Conf: {result['confidence']}/{self.config['confidence_max']}"
        ratio_text = f"Ratio: {result['obstacle_ratio']:.3f}"
        
        cv2.putText(vis_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 
                   (0, 0, 255) if result['obstacle_detected'] else (0, 255, 0), 
                   2)
        cv2.putText(vis_frame, conf_text, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis_frame, ratio_text, (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return vis_frame

def process_dataset(dataset_path, output_path=None):
    """Process a dataset of images and save/display results"""
    # Create detector
    detector = ImprovedObstacleDetector()
    
    # Get all images
    image_paths = sorted(glob.glob(os.path.join(dataset_path, "*.jpg")))
    
    # Create output directory if needed
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    for img_path in image_paths:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading {img_path}")
            continue
            
        # Process frame
        result = detector.process_frame(img)
        
        # Create visualization
        vis_img = detector.visualize_detection(img, result)
        
        # Save or display
        if output_path:
            base_name = os.path.basename(img_path)
            output_file = os.path.join(output_path, f"processed_{base_name}")
            cv2.imwrite(output_file, vis_img)
            print(f"Processed {base_name} -> {output_file}")
        else:
            # Display
            cv2.imshow("Detection Result", vis_img)
            key = cv2.waitKey(0)
            if key == 27:  # ESC key
                break
    
    cv2.destroyAllWindows()

def main():
    """Main function to run the detector on a dataset or webcam"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Improved Obstacle Detection")
    parser.add_argument("--dataset", help="Path to dataset folder")
    parser.add_argument("--output", help="Path to output folder")
    parser.add_argument("--webcam", action="store_true", help="Use webcam instead of dataset")
    parser.add_argument("--webcam_id", type=int, default=0, help="Webcam ID to use")
    
    args = parser.parse_args()
    
    if args.webcam:
        # Create detector
        detector = ImprovedObstacleDetector()
        
        # Open webcam
        cap = cv2.VideoCapture(args.webcam_id)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            result = detector.process_frame(frame)
            
            # Create visualization
            vis_frame = detector.visualize_detection(frame, result)
            
            # Display
            cv2.imshow("Obstacle Detection", vis_frame)
            
            # Exit on ESC
            key = cv2.waitKey(1)
            if key == 27:
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    elif args.dataset:
        process_dataset(args.dataset, args.output)
    
    else:
        print("Please specify either --dataset or --webcam")

if __name__ == "__main__":
    main()
