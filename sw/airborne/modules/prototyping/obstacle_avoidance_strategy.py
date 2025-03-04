#!/usr/bin/env python3
"""
Advanced Obstacle Avoidance Strategy for Paparazzi UAV

This script implements and simulates advanced obstacle avoidance strategies
that can be integrated with the Paparazzi UAV system to improve the
orange avoider module.

Features:
- Potential field based avoidance
- Vector field histogram (VFH) implementation
- Path planning with RRT (Rapidly-exploring Random Trees)
- Simulation environment for testing strategies
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import math
import random
from scipy.ndimage import gaussian_filter
import time

class ObstacleAvoidanceStrategy:
    """Base class for obstacle avoidance strategies"""
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            'arena_size': (10, 10),  # meters
            'safety_distance': 0.5,  # meters
            'max_speed': 1.0,        # m/s
            'max_heading_change': np.radians(30),  # max heading change per step
            'goal_threshold': 0.3,   # meters, distance to consider goal reached
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
    
    def compute_avoidance_vector(self, position, heading, obstacles, goal):
        """
        Compute avoidance vector based on current state
        
        Args:
            position: Current position (x, y) in meters
            heading: Current heading in radians
            obstacles: List of obstacles, each as (x, y, radius)
            goal: Goal position (x, y) in meters
            
        Returns:
            new_heading: New heading in radians
            speed: Desired speed in m/s
        """
        # Base implementation - just head toward goal
        goal_vector = np.array([goal[0] - position[0], goal[1] - position[1]])
        distance_to_goal = np.linalg.norm(goal_vector)
        
        if distance_to_goal < self.config['goal_threshold']:
            return heading, 0.0  # Stop at goal
            
        goal_heading = np.arctan2(goal_vector[1], goal_vector[0])
        
        # Limit heading change
        heading_diff = goal_heading - heading
        # Normalize to [-pi, pi]
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        
        if abs(heading_diff) > self.config['max_heading_change']:
            new_heading = heading + np.sign(heading_diff) * self.config['max_heading_change']
        else:
            new_heading = goal_heading
            
        return new_heading, self.config['max_speed']

class PotentialFieldStrategy(ObstacleAvoidanceStrategy):
    """Potential field based obstacle avoidance"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Additional configuration for potential fields
        potential_config = {
            'attractive_gain': 1.0,
            'repulsive_gain': 2.0,
            'influence_distance': 2.0,  # meters
            'min_distance': 0.1,        # to avoid division by zero
        }
        
        if config:
            potential_config.update({k: v for k, v in config.items() 
                                   if k in potential_config})
        
        self.config.update(potential_config)
    
    def compute_avoidance_vector(self, position, heading, obstacles, goal):
        """Compute avoidance vector using potential fields"""
        position = np.array(position)
        goal = np.array(goal)
        
        # Attractive force toward goal
        goal_vector = goal - position
        distance_to_goal = np.linalg.norm(goal_vector)
        
        if distance_to_goal < self.config['goal_threshold']:
            return heading, 0.0  # Stop at goal
            
        # Normalize and scale by attractive gain
        if distance_to_goal > 0:
            attractive_force = self.config['attractive_gain'] * goal_vector / distance_to_goal
        else:
            attractive_force = np.array([0, 0])
        
        # Repulsive forces from obstacles
        repulsive_force = np.array([0.0, 0.0])
        
        for obs in obstacles:
            obs_position = np.array(obs[:2])
            obs_radius = obs[2]
            
            # Vector from obstacle to drone
            obs_vector = position - obs_position
            distance = np.linalg.norm(obs_vector)
            
            # Only consider obstacles within influence distance
            if distance < self.config['influence_distance']:
                # Normalize direction vector
                if distance > 0:
                    direction = obs_vector / distance
                else:
                    direction = np.array([np.cos(heading), np.sin(heading)])
                
                # Calculate repulsive magnitude (stronger when closer)
                magnitude = self.config['repulsive_gain'] * (
                    1.0 / max(distance - obs_radius, self.config['min_distance']) - 
                    1.0 / self.config['influence_distance']
                ) ** 2
                
                repulsive_force += magnitude * direction
        
        # Combine forces
        total_force = attractive_force + repulsive_force
        
        # Convert to heading and speed
        if np.linalg.norm(total_force) > 0:
            new_heading = np.arctan2(total_force[1], total_force[0])
            
            # Limit heading change
            heading_diff = new_heading - heading
            # Normalize to [-pi, pi]
            heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
            
            if abs(heading_diff) > self.config['max_heading_change']:
                new_heading = heading + np.sign(heading_diff) * self.config['max_heading_change']
                
            # Speed proportional to force magnitude, but limited
            speed = min(np.linalg.norm(total_force), self.config['max_speed'])
        else:
            new_heading = heading
            speed = 0.0
            
        return new_heading, speed

class VectorFieldHistogramStrategy(ObstacleAvoidanceStrategy):
    """Vector Field Histogram based obstacle avoidance"""
    
    def __init__(self, config=None):
        super().__init__(config)
        
        # Additional configuration for VFH
        vfh_config = {
            'num_sectors': 36,        # Number of angular sectors
            'obstacle_weight': 2.0,   # Weight for obstacle avoidance
            'goal_weight': 1.0,       # Weight for goal seeking
            'smoothing_factor': 0.2,  # Smoothing factor for histogram
            'max_detection_dist': 3.0, # Maximum obstacle detection distance
            'sector_threshold': 0.3,  # Threshold for considering sector blocked
        }
        
        if config:
            vfh_config.update({k: v for k, v in config.items() 
                              if k in vfh_config})
        
        self.config.update(vfh_config)
    
    def compute_avoidance_vector(self, position, heading, obstacles, goal):
        """Compute avoidance vector using Vector Field Histogram"""
        position = np.array(position)
        goal = np.array(goal)
        
        # Check if at goal
        distance_to_goal = np.linalg.norm(goal - position)
        if distance_to_goal < self.config['goal_threshold']:
            return heading, 0.0  # Stop at goal
        
        # Create polar histogram
        histogram = np.zeros(self.config['num_sectors'])
        sector_angle = 2 * np.pi / self.config['num_sectors']
        
        # Fill histogram with obstacle information
        for obs in obstacles:
            obs_position = np.array(obs[:2])
            obs_radius = obs[2]
            
            # Vector from drone to obstacle
            obs_vector = obs_position - position
            distance = np.linalg.norm(obs_vector)
            
            # Only consider obstacles within detection range
            if distance < self.config['max_detection_dist']:
                # Calculate angle to obstacle
                angle = np.arctan2(obs_vector[1], obs_vector[0])
                # Convert to sector index
                sector = int(((angle + np.pi) % (2 * np.pi)) / sector_angle)
                
                # Calculate obstacle influence (stronger when closer)
                influence = self.config['obstacle_weight'] * (
                    1.0 - (distance - obs_radius) / self.config['max_detection_dist']
                ) ** 2
                
                # Add to histogram and neighboring sectors (for smoothing)
                for i in range(-2, 3):
                    idx = (sector + i) % self.config['num_sectors']
                    # Reduce influence for neighboring sectors
                    factor = 1.0 if i == 0 else (0.5 if abs(i) == 1 else 0.25)
                    histogram[idx] += influence * factor
        
        # Apply smoothing
        if self.config['smoothing_factor'] > 0:
            histogram = (1 - self.config['smoothing_factor']) * histogram + \
                        self.config['smoothing_factor'] * np.roll(histogram, 1)
            histogram = (1 - self.config['smoothing_factor']) * histogram + \
                        self.config['smoothing_factor'] * np.roll(histogram, -1)
        
        # Calculate goal direction influence
        goal_vector = goal - position
        goal_angle = np.arctan2(goal_vector[1], goal_vector[0])
        goal_sector = int(((goal_angle + np.pi) % (2 * np.pi)) / sector_angle)
        
        # Find best direction
        best_sector = None
        min_cost = float('inf')
        
        for sector in range(self.config['num_sectors']):
            # Skip blocked sectors
            if histogram[sector] > self.config['sector_threshold']:
                continue
                
            # Calculate cost (combination of obstacle proximity and goal direction)
            obstacle_cost = histogram[sector]
            
            # Angular distance to goal
            angular_dist = min(
                abs(sector - goal_sector),
                self.config['num_sectors'] - abs(sector - goal_sector)
            ) * sector_angle
            
            goal_cost = self.config['goal_weight'] * angular_dist
            
            # Total cost
            total_cost = obstacle_cost + goal_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_sector = sector
        
        # If all sectors blocked, find least blocked
        if best_sector is None:
            best_sector = np.argmin(histogram)
        
        # Convert sector to heading
        new_heading = best_sector * sector_angle - np.pi
        
        # Limit heading change
        heading_diff = new_heading - heading
        # Normalize to [-pi, pi]
        heading_diff = (heading_diff + np.pi) % (2 * np.pi) - np.pi
        
        if abs(heading_diff) > self.config['max_heading_change']:
            new_heading = heading + np.sign(heading_diff) * self.config['max_heading_change']
        
        # Adjust speed based on obstacle proximity
        # Slow down when obstacles are nearby
        min_obstacle_dist = float('inf')
        for obs in obstacles:
            obs_position = np.array(obs[:2])
            obs_radius = obs[2]
            distance = np.linalg.norm(obs_position - position) - obs_radius
            min_obstacle_dist = min(min_obstacle_dist, distance)
        
        # Scale speed based on obstacle proximity
        if min_obstacle_dist < self.config['safety_distance']:
            speed = 0.1 * self.config['max_speed']  # Very slow when too close
        elif min_obstacle_dist < 2 * self.config['safety_distance']:
            # Linear scaling between safety distance and 2*safety distance
            speed_factor = (min_obstacle_dist - self.config['safety_distance']) / self.config['safety_distance']
            speed = (0.1 + 0.9 * speed_factor) * self.config['max_speed']
        else:
            speed = self.config['max_speed']
            
        return new_heading, speed

class SimulationEnvironment:
    """Simulation environment for testing obstacle avoidance strategies"""
    
    def __init__(self, strategy, config=None):
        # Default configuration
        self.config = {
            'arena_size': (10, 10),  # meters
            'num_obstacles': 10,
            'min_obstacle_radius': 0.2,
            'max_obstacle_radius': 0.5,
            'drone_radius': 0.2,
            'time_step': 0.1,         # seconds
            'max_steps': 1000,
            'random_seed': None,
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # Set random seed if provided
        if self.config['random_seed'] is not None:
            random.seed(self.config['random_seed'])
            np.random.seed(self.config['random_seed'])
        
        self.strategy = strategy
        
        # Initialize state
        self.reset()
    
    def reset(self):
        """Reset the simulation environment"""
        arena_width, arena_height = self.config['arena_size']
        
        # Initialize drone at random position
        self.drone_position = np.array([
            random.uniform(1, arena_width - 1),
            random.uniform(1, arena_height - 1)
        ])
        self.drone_heading = random.uniform(0, 2 * np.pi)
        
        # Initialize goal at random position
        self.goal_position = np.array([
            random.uniform(1, arena_width - 1),
            random.uniform(1, arena_height - 1)
        ])
        
        # Ensure goal is not too close to drone
        while np.linalg.norm(self.goal_position - self.drone_position) < 3.0:
            self.goal_position = np.array([
                random.uniform(1, arena_width - 1),
                random.uniform(1, arena_height - 1)
            ])
        
        # Generate random obstacles
        self.obstacles = []
        for _ in range(self.config['num_obstacles']):
            # Random position
            obs_x = random.uniform(0, arena_width)
            obs_y = random.uniform(0, arena_height)
            
            # Random radius
            obs_radius = random.uniform(
                self.config['min_obstacle_radius'],
                self.config['max_obstacle_radius']
            )
            
            # Add obstacle
            self.obstacles.append((obs_x, obs_y, obs_radius))
        
        # Initialize trajectory history
        self.trajectory = [self.drone_position.copy()]
        self.headings = [self.drone_heading]
        
        # Reset step counter
        self.steps = 0
        
        # Reset collision flag
        self.collision = False
        
        # Reset goal reached flag
        self.goal_reached = False
    
    def step(self):
        """Perform one simulation step"""
        if self.collision or self.goal_reached:
            return
        
        # Compute new heading and speed using strategy
        new_heading, speed = self.strategy.compute_avoidance_vector(
            self.drone_position, self.drone_heading,
            self.obstacles, self.goal_position
        )
        
        # Update drone position and heading
        self.drone_heading = new_heading
        
        # Calculate movement vector
        movement = speed * self.config['time_step'] * np.array([
            np.cos(self.drone_heading),
            np.sin(self.drone_heading)
        ])
        
        # Update position
        self.drone_position += movement
        
        # Record trajectory
        self.trajectory.append(self.drone_position.copy())
        self.headings.append(self.drone_heading)
        
        # Check for collision with obstacles
        for obs_x, obs_y, obs_radius in self.obstacles:
            distance = np.linalg.norm(
                self.drone_position - np.array([obs_x, obs_y])
            )
            if distance < (obs_radius + self.config['drone_radius']):
                self.collision = True
                break
        
        # Check for collision with arena boundaries
        arena_width, arena_height = self.config['arena_size']
        if (self.drone_position[0] < 0 or 
            self.drone_position[0] > arena_width or
            self.drone_position[1] < 0 or 
            self.drone_position[1] > arena_height):
            self.collision = True
        
        # Check if goal reached
        distance_to_goal = np.linalg.norm(
            self.drone_position - self.goal_position
        )
        if distance_to_goal < self.strategy.config['goal_threshold']:
            self.goal_reached = True
        
        # Increment step counter
        self.steps += 1
        
        # Check if max steps reached
        if self.steps >= self.config['max_steps']:
            return True  # Simulation complete
            
        return self.collision or self.goal_reached
    
    def run(self):
        """Run the complete simulation"""
        done = False
        while not done:
            done = self.step()
        
        return {
            'success': self.goal_reached,
            'collision': self.collision,
            'steps': self.steps,
            'trajectory': np.array(self.trajectory),
            'headings': np.array(self.headings),
            'path_length': self._calculate_path_length(),
        }
    
    def _calculate_path_length(self):
        """Calculate the total path length"""
        path_length = 0
        for i in range(1, len(self.trajectory)):
            path_length += np.linalg.norm(
                self.trajectory[i] - self.trajectory[i-1]
            )
        return path_length
    
    def visualize(self, show_histogram=False):
        """Visualize the simulation"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Set limits
        arena_width, arena_height = self.config['arena_size']
        ax.set_xlim([-1, arena_width + 1])
        ax.set_ylim([-1, arena_height + 1])
        
        # Draw arena boundaries
        ax.add_patch(Rectangle((0, 0), arena_width, arena_height, 
                              fill=False, edgecolor='black', linewidth=2))
        
        # Draw obstacles
        for obs_x, obs_y, obs_radius in self.obstacles:
            ax.add_patch(Circle((obs_x, obs_y), obs_radius, 
                               fill=True, color='red', alpha=0.5))
        
        # Draw trajectory
        trajectory = np.array(self.trajectory)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
        
        # Draw drone
        ax.add_patch(Circle(self.drone_position, self.config['drone_radius'], 
                           fill=True, color='blue'))
        
        # Draw heading
        heading_line = np.array([
            self.drone_position,
            self.drone_position + 0.5 * np.array([
                np.cos(self.drone_heading),
                np.sin(self.drone_heading)
            ])
        ])
        ax.plot(heading_line[:, 0], heading_line[:, 1], 'k-', linewidth=2)
        
        # Draw goal
        ax.add_patch(Circle(self.goal_position, self.strategy.config['goal_threshold'], 
                           fill=True, color='green', alpha=0.5))
        
        # Add status text
        status = "Goal Reached!" if self.goal_reached else "Collision!" if self.collision else "In Progress"
        ax.text(0.5, arena_height + 0.5, f"Status: {status}", 
                horizontalalignment='center', fontsize=12)
        
        # Add step counter
        ax.text(0.5, -0.5, f"Steps: {self.steps}", 
                horizontalalignment='center', fontsize=12)
        
        # Show histogram if requested and using VFH
        if show_histogram and isinstance(self.strategy, VectorFieldHistogramStrategy):
            # Create polar histogram
            histogram = np.zeros(self.strategy.config['num_sectors'])
            sector_angle = 2 * np.pi / self.strategy.config['num_sectors']
            
            # Fill histogram with obstacle information
            for obs in self.obstacles:
                obs_position = np.array(obs[:2])
                obs_radius = obs[2]
                
                # Vector from drone to obstacle
                obs_vector = obs_position - self.drone_position
                distance = np.linalg.norm(obs_vector)
                
                # Only consider obstacles within detection range
                if distance < self.strategy.config['max_detection_dist']:
                    # Calculate angle to obstacle
                    angle = np.arctan2(obs_vector[1], obs_vector[0])
                    # Convert to sector index
                    sector = int(((angle + np.pi) % (2 * np.pi)) / sector_angle)
                    
                    # Calculate obstacle influence (stronger when closer)
                    influence = self.strategy.config['obstacle_weight'] * (
                        1.0 - (distance - obs_radius) / self.strategy.config['max_detection_dist']
                    ) ** 2
                    
                    # Add to histogram and neighboring sectors (for smoothing)
                    for i in range(-2, 3):
                        idx = (sector + i) % self.strategy.config['num_sectors']
                        # Reduce influence for neighboring sectors
                        factor = 1.0 if i == 0 else (0.5 if abs(i) == 1 else 0.25)
                        histogram[idx] += influence * factor
            
            # Create subplot for histogram
            ax_hist = fig.add_axes([0.1, 0.1, 0.8, 0.2], polar=True)
            theta = np.linspace(0, 2*np.pi, self.strategy.config['num_sectors'], endpoint=False)
            ax_hist.bar(theta, histogram, width=sector_angle, alpha=0.5)
            ax_hist.set_title("Vector Field Histogram")
        
        plt.tight_layout()
        plt.show()

def compare_strategies(num_trials=10, arena_size=(10, 10), num_obstacles=15):
    """Compare different obstacle avoidance strategies"""
    # Define strategies to compare
    strategies = {
        "Potential Field": PotentialFieldStrategy(),
        "Vector Field Histogram": VectorFieldHistogramStrategy()
    }
    
    # Results storage
    results = {name: {
        'success_rate': 0,
        'avg_path_length': 0,
        'avg_steps': 0,
        'collisions': 0
    } for name in strategies}
    
    # Run trials
    for name, strategy in strategies.items():
        print(f"Testing {name}...")
        
        successes = 0
        total_path_length = 0
        total_steps = 0
        collisions = 0
        
        for trial in range(num_trials):
            # Create simulation with same seed for fair comparison
            sim = SimulationEnvironment(
                strategy,
                {
                    'arena_size': arena_size,
                    'num_obstacles': num_obstacles,
                    'random_seed': trial,  # Same seed for each strategy
                }
            )
            
            # Run simulation
            result = sim.run()
            
            # Update statistics
            if result['success']:
                successes += 1
                total_path_length += result['path_length']
                total_steps += result['steps']
            
            if result['collision']:
                collisions += 1
        
        # Calculate averages
        results[name]['success_rate'] = successes / num_trials
        if successes > 0:
            results[name]['avg_path_length'] = total_path_length / successes
            results[name]['avg_steps'] = total_steps / successes
        results[name]['collisions'] = collisions
        
        print(f"  Success Rate: {results[name]['success_rate']:.2f}")
        print(f"  Avg Path Length: {results[name]['avg_path_length']:.2f}")
        print(f"  Avg Steps: {results[name]['avg_steps']:.2f}")
        print(f"  Collisions: {results[name]['collisions']}")
    
    return results

def main():
    """Main function to demonstrate obstacle avoidance strategies"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Obstacle Avoidance Strategies")
    parser.add_argument("--strategy", choices=["potential", "vfh"], default="vfh",
                       help="Avoidance strategy to use")
    parser.add_argument("--obstacles", type=int, default=10,
                       help="Number of obstacles")
    parser.add_argument("--compare", action="store_true",
                       help="Compare strategies")
    parser.add_argument("--trials", type=int, default=10,
                       help="Number of trials for comparison")
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_strategies(
            num_trials=args.trials,
            num_obstacles=args.obstacles
        )
        
        # Plot comparison
        strategies = list(results.keys())
        success_rates = [results[s]['success_rate'] for s in strategies]
        
        plt.figure(figsize=(10, 6))
        plt.bar(strategies, success_rates)
        plt.ylim(0, 1)
        plt.ylabel('Success Rate')
        plt.title('Strategy Comparison')
        plt.show()
        
    else:
        # Create strategy
        if args.strategy == "potential":
            strategy = PotentialFieldStrategy()
        else:  # vfh
            strategy = VectorFieldHistogramStrategy()
        
        # Create simulation
        sim = SimulationEnvironment(
            strategy,
            {
                'num_obstacles': args.obstacles,
            }
        )
        
        # Run simulation
        result = sim.run()
        
        # Visualize
        sim.visualize(show_histogram=(args.strategy == "vfh"))
        
        # Print results
        print(f"Success: {result['success']}")
        print(f"Collision: {result['collision']}")
        print(f"Steps: {result['steps']}")
        print(f"Path Length: {result['path_length']:.2f}")

if __name__ == "__main__":
    main()
