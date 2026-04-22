"""
RGBD Camera interface for MuJoCo.
"""
import numpy as np
import mujoco


class RGBDCamera:
    """RGBD camera for capturing color and depth images from MuJoCo."""
    
    def __init__(self, model, data, camera_name, width=640, height=480):
        """
        Initialize RGBD camera.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            camera_name: Name of the camera in the XML
            width: Image width
            height: Image height
        """
        self.model = model
        self.data = data
        self.camera_name = camera_name
        self.width = width
        self.height = height
        
        # Get camera ID
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        
        # Create renderer
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        
        # Compute camera intrinsics
        self.intrinsics = self._compute_intrinsics()
        
        # Get camera extrinsics (will be updated each frame)
        self.extrinsics = None
    
    def _compute_intrinsics(self):
        """Compute camera intrinsic matrix."""
        # Get camera FOV
        fovy = self.model.cam_fovy[self.camera_id]
        
        # Convert to radians
        fovy_rad = np.deg2rad(fovy)
        
        # Compute focal length
        fy = self.height / (2 * np.tan(fovy_rad / 2))
        fx = fy  # Assume square pixels
        
        # Principal point
        cx = self.width / 2
        cy = self.height / 2
        
        # Intrinsic matrix
        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        
        return intrinsics
    
    def get_extrinsics(self):
        """Get camera extrinsics (transformation from world to camera)."""
        # Get camera pose
        camera_pos = self.data.cam_xpos[self.camera_id].copy()
        camera_mat = self.data.cam_xmat[self.camera_id].copy().reshape(3, 3)
        
        # Build transformation matrix (world to camera)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = camera_mat
        extrinsics[:3, 3] = camera_pos
        
        return extrinsics
    
    def render(self):
        """
        Render RGB and depth images.
        
        Returns:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
        """
        # Update scene
        self.renderer.update_scene(self.data, camera=self.camera_name)
        
        # Render RGB
        rgb = self.renderer.render()
        
        # Render depth (compatible with different mujoco Python APIs).
        try:
            depth = self.renderer.render(depth=True)
        except TypeError:
            if hasattr(self.renderer, "enable_depth_rendering"):
                self.renderer.enable_depth_rendering()
                depth = self.renderer.render()
                self.renderer.disable_depth_rendering()
            else:
                raise
        
        # Update extrinsics
        self.extrinsics = self.get_extrinsics()

        return rgb, depth

    def capture(self, timestamp=None):
        """
        Capture one synchronized RGBD frame and camera parameters.

        Args:
            timestamp: Optional frame timestamp from simulator time.

        Returns:
            Dict with rgb, depth, intrinsics, extrinsics, timestamp.
        """
        rgb, depth = self.render()
        return {
            "rgb": rgb,
            "depth": depth,
            "intrinsics": self.intrinsics.copy(),
            "extrinsics": self.extrinsics.copy() if self.extrinsics is not None else None,
            "timestamp": timestamp
        }
    
    def get_camera_params(self):
        """Get camera parameters."""
        return {
            'intrinsics': self.intrinsics,
            'extrinsics': self.extrinsics,
            'width': self.width,
            'height': self.height
        }

    def close(self):
        """Release renderer resources explicitly to avoid teardown crashes."""
        if hasattr(self, "renderer") and self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
