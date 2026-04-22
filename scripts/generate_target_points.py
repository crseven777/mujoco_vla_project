"""
Generate target points for end-effector tracking.
"""
import numpy as np
import argparse


class TargetGenerator:
    """Generator for different types of target trajectories."""
    
    @staticmethod
    def generate_static_target(position, noise_std=0.0):
        """
        Generate a static target point.
        
        Args:
            position: [x, y, z] target position
            noise_std: Standard deviation of Gaussian noise to add
        
        Returns:
            Target position as numpy array
        """
        target = np.array(position, dtype=np.float32)
        if noise_std > 0:
            target += np.random.normal(0, noise_std, size=3)
        return target
    
    @staticmethod
    def generate_multi_point_trajectory(points, hold_steps=100, transition_steps=50):
        """
        Generate a trajectory with multiple points.
        
        Args:
            points: List of [x, y, z] positions
            hold_steps: Number of steps to hold at each point
            transition_steps: Number of steps for linear interpolation between points
        
        Returns:
            List of target positions forming the trajectory
        """
        trajectory = []
        
        for i in range(len(points)):
            # Hold at current point
            for _ in range(hold_steps):
                trajectory.append(np.array(points[i], dtype=np.float32))
            
            # Transition to next point (if not last)
            if i < len(points) - 1:
                for t in range(transition_steps):
                    alpha = t / transition_steps
                    interp = (1 - alpha) * np.array(points[i]) + alpha * np.array(points[i + 1])
                    trajectory.append(interp.astype(np.float32))
        
        return trajectory
    
    @staticmethod
    def generate_circular_trajectory(center, radius, num_points=200, height=1.2):
        """
        Generate a circular trajectory.
        
        Args:
            center: [x, y] center position (horizontal plane)
            radius: Circle radius
            num_points: Number of points in the trajectory
            height: Fixed z height
        
        Returns:
            List of target positions forming a circle
        """
        trajectory = []
        for i in range(num_points):
            angle = 2 * np.pi * i / num_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = height
            trajectory.append(np.array([x, y, z], dtype=np.float32))
        return trajectory
    
    @staticmethod
    def generate_square_trajectory(center, side_length, num_points_per_side=50, height=1.2):
        """
        Generate a square trajectory.
        
        Args:
            center: [x, y] center position
            side_length: Length of each side
            num_points_per_side: Number of points per side
            height: Fixed z height
        
        Returns:
            List of target positions forming a square
        """
        half_side = side_length / 2
        corners = [
            [center[0] - half_side, center[1] - half_side],
            [center[0] + half_side, center[1] - half_side],
            [center[0] + half_side, center[1] + half_side],
            [center[0] - half_side, center[1] + half_side],
            [center[0] - half_side, center[1] - half_side]  # Close the loop
        ]
        
        trajectory = []
        for i in range(len(corners) - 1):
            for t in range(num_points_per_side):
                alpha = t / num_points_per_side
                x = (1 - alpha) * corners[i][0] + alpha * corners[i + 1][0]
                y = (1 - alpha) * corners[i][1] + alpha * corners[i + 1][1]
                trajectory.append(np.array([x, y, height], dtype=np.float32))
        
        return trajectory


def main():
    parser = argparse.ArgumentParser(description="Generate target points for demonstration")
    parser.add_argument("--type", type=str, default="static", 
                       choices=["static", "multi", "circle", "square"],
                       help="Type of target trajectory")
    parser.add_argument("--output", type=str, default="target_points.npy",
                       help="Output file path")
    args = parser.parse_args()
    
    generator = TargetGenerator()
    
    if args.type == "static":
        # Static target in front of right hand
        target = generator.generate_static_target([0.3, -0.3, 1.2])
        targets = [target]
        print(f"Generated static target: {target}")
        
    elif args.type == "multi":
        # Multiple points in workspace
        points = [
            [0.3, -0.3, 1.2],
            [0.4, -0.2, 1.3],
            [0.3, -0.4, 1.1],
            [0.2, -0.3, 1.2]
        ]
        targets = generator.generate_multi_point_trajectory(points, hold_steps=100, transition_steps=50)
        print(f"Generated multi-point trajectory with {len(targets)} points")
        
    elif args.type == "circle":
        # Circular trajectory
        targets = generator.generate_circular_trajectory(
            center=[0.3, -0.3], 
            radius=0.1, 
            num_points=200,
            height=1.2
        )
        print(f"Generated circular trajectory with {len(targets)} points")
        
    elif args.type == "square":
        # Square trajectory
        targets = generator.generate_square_trajectory(
            center=[0.3, -0.3],
            side_length=0.2,
            num_points_per_side=50,
            height=1.2
        )
        print(f"Generated square trajectory with {len(targets)} points")
    
    # Save targets
    np.save(args.output, np.array(targets))
    print(f"Saved targets to {args.output}")


if __name__ == "__main__":
    main()