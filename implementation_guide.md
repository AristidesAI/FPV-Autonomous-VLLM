# Micro-UAV Swarm Implementation Guide

## Quick Start Guide

This guide provides practical, step-by-step instructions for implementing the software components of your micro-UAV swarm system.

## 1. Ground Station Setup

### 1.1 Development Environment

```bash
# Create Python virtual environment
python3 -m venv drone_swarm_env
source drone_swarm_env/bin/activate  # Linux/Mac
# or
drone_swarm_env\Scripts\activate  # Windows

# Install core dependencies
pip install -r requirements.txt
```

### 1.2 Core Requirements File

Create `requirements.txt`:
```txt
# Computer Vision & ML
opencv-python==4.8.1
ultralytics==8.0.200  # YOLOv8
torch==2.1.0
torchvision==0.16.0
tensorflow==2.14.0  # Alternative to PyTorch
onnxruntime==1.16.0

# SLAM & Mapping
opencv-contrib-python==4.8.1  # Extra modules for SLAM
pyrealsense2==2.54.1  # If using Intel RealSense
open3d==0.17.0  # Point cloud processing

# Drone Control
pymavlink==2.4.40
dronekit==2.9.2
pyserial==3.5  # For MSP protocol

# Video Processing
numpy==1.24.3
Pillow==10.0.0
scikit-image==0.21.0
imutils==0.5.4

# Communication
pyzmq==25.1.0
grpcio==1.59.0
websocket-client==1.6.3

# Data Management
pandas==2.0.3
matplotlib==3.7.2
tensorboard==2.14.0
```

## 2. Video Reception Pipeline

### 2.1 Multi-Stream Video Capture

```python
# video_receiver.py
import cv2
import threading
import queue
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class DroneStream:
    drone_id: str
    device_id: int
    resolution: tuple = (1920, 1080)
    fps: int = 30

class MultiStreamReceiver:
    def __init__(self, drone_configs: list):
        self.streams = {}
        self.frame_queues = {}
        self.threads = {}
        
        for config in drone_configs:
            self.add_stream(config)
    
    def add_stream(self, config: DroneStream):
        """Add a new drone video stream"""
        cap = cv2.VideoCapture(config.device_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.resolution[1])
        cap.set(cv2.CAP_PROP_FPS, config.fps)
        
        self.streams[config.drone_id] = cap
        self.frame_queues[config.drone_id] = queue.Queue(maxsize=10)
        
        # Start capture thread
        thread = threading.Thread(
            target=self._capture_loop,
            args=(config.drone_id,)
        )
        thread.daemon = True
        thread.start()
        self.threads[config.drone_id] = thread
    
    def _capture_loop(self, drone_id: str):
        """Continuous capture loop for a single stream"""
        cap = self.streams[drone_id]
        q = self.frame_queues[drone_id]
        
        while True:
            ret, frame = cap.read()
            if ret:
                # Apply analog video denoising
                frame = self._denoise_analog(frame)
                
                # Add to queue (drop old frames if full)
                if q.full():
                    q.get()
                q.put(frame)
    
    def _denoise_analog(self, frame):
        """Remove analog video noise"""
        # Temporal denoising for analog artifacts
        frame = cv2.fastNlMeansDenoisingColored(
            frame, None, 10, 10, 7, 21
        )
        # Reduce interlacing artifacts
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        return frame
    
    def get_frame(self, drone_id: str) -> Optional[np.ndarray]:
        """Get latest frame from drone"""
        if drone_id in self.frame_queues:
            try:
                return self.frame_queues[drone_id].get_nowait()
            except queue.Empty:
                return None
        return None
    
    def get_all_frames(self) -> Dict[str, np.ndarray]:
        """Get frames from all drones"""
        frames = {}
        for drone_id in self.streams.keys():
            frame = self.get_frame(drone_id)
            if frame is not None:
                frames[drone_id] = frame
        return frames

# Usage example
if __name__ == "__main__":
    # Configure multiple drone streams
    drone_configs = [
        DroneStream("drone_1", device_id=0),  # USB capture device 0
        DroneStream("drone_2", device_id=1),  # USB capture device 1
        DroneStream("drone_3", device_id=2),  # USB capture device 2
    ]
    
    receiver = MultiStreamReceiver(drone_configs)
    
    # Process frames
    while True:
        frames = receiver.get_all_frames()
        for drone_id, frame in frames.items():
            cv2.imshow(f"Drone {drone_id}", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
```

## 3. Real-Time Object Detection

### 3.1 YOLOv8 Integration

```python
# detection_engine.py
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple
import time

class DetectionEngine:
    def __init__(self, model_path: str = 'yolov8n.pt', 
                 device: str = 'cuda'):
        """
        Initialize YOLO detection engine
        Args:
            model_path: Path to YOLO weights
            device: 'cuda' for GPU, 'cpu' for CPU
        """
        self.device = device
        self.model = YOLO(model_path)
        
        # Move model to GPU if available
        if device == 'cuda' and torch.cuda.is_available():
            self.model.to('cuda')
        
        # Warm up model
        self._warmup()
    
    def _warmup(self):
        """Warm up GPU with dummy inference"""
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)
    
    def detect(self, frame: np.ndarray, 
               conf_threshold: float = 0.5) -> List[Dict]:
        """
        Run detection on single frame
        Returns list of detections with format:
        [{'bbox': [x1,y1,x2,y2], 'class': str, 'conf': float}]
        """
        results = self.model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'class': self.model.names[int(box.cls)],
                        'conf': float(box.conf)
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: Dict[str, np.ndarray],
                    conf_threshold: float = 0.5) -> Dict[str, List]:
        """
        Run detection on multiple frames efficiently
        """
        # Stack frames for batch processing
        drone_ids = list(frames.keys())
        frame_batch = [frames[did] for did in drone_ids]
        
        # Batch inference
        results = self.model(frame_batch, conf=conf_threshold, 
                           verbose=False, stream=True)
        
        # Parse results
        all_detections = {}
        for drone_id, r in zip(drone_ids, results):
            detections = []
            if r.boxes is not None:
                for box in r.boxes:
                    detection = {
                        'bbox': box.xyxy[0].cpu().numpy().tolist(),
                        'class': self.model.names[int(box.cls)],
                        'conf': float(box.conf),
                        'track_id': int(box.id) if box.id is not None else -1
                    }
                    detections.append(detection)
            all_detections[drone_id] = detections
        
        return all_detections
    
    def draw_detections(self, frame: np.ndarray, 
                        detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on frame"""
        for det in detections:
            x1, y1, x2, y2 = [int(x) for x in det['bbox']]
            label = f"{det['class']} {det['conf']:.2f}"
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, 1)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 4),
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame

# Specialized drone detection model
class DroneTargetDetector(DetectionEngine):
    def __init__(self, model_path: str = 'custom_drone_model.pt'):
        super().__init__(model_path)
        
        # Define priority targets
        self.priority_classes = ['person', 'car', 'truck', 'airplane']
    
    def detect_targets(self, frame: np.ndarray) -> Dict:
        """Detect and prioritize targets"""
        detections = self.detect(frame)
        
        # Filter and prioritize
        targets = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        }
        
        for det in detections:
            if det['class'] in self.priority_classes:
                if det['conf'] > 0.8:
                    targets['high_priority'].append(det)
                elif det['conf'] > 0.6:
                    targets['medium_priority'].append(det)
                else:
                    targets['low_priority'].append(det)
        
        return targets
```

## 4. Distributed SLAM Implementation

### 4.1 Visual SLAM Wrapper

```python
# slam_processor.py
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import threading
import queue

class VisualSLAM:
    def __init__(self, camera_matrix: np.ndarray, 
                 dist_coeffs: np.ndarray):
        """
        Initialize Visual SLAM processor
        Args:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.K = camera_matrix
        self.dist = dist_coeffs
        
        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Map storage
        self.keyframes = []
        self.map_points = []
        self.current_pose = np.eye(4)
    
    def process_frame(self, frame: np.ndarray, 
                      timestamp: float) -> Dict:
        """Process single frame for SLAM"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Extract features
        kp, des = self.orb.detectAndCompute(gray, None)
        
        # Match with last keyframe if exists
        if len(self.keyframes) > 0:
            matches = self._match_features(
                des, self.keyframes[-1]['descriptors']
            )
            
            if len(matches) > 30:
                # Estimate pose
                pose = self._estimate_pose(
                    kp, self.keyframes[-1]['keypoints'], 
                    matches
                )
                if pose is not None:
                    self.current_pose = pose @ self.current_pose
        
        # Check if new keyframe needed
        if self._is_keyframe(kp, des):
            self.keyframes.append({
                'timestamp': timestamp,
                'keypoints': kp,
                'descriptors': des,
                'pose': self.current_pose.copy()
            })
        
        return {
            'pose': self.current_pose,
            'num_features': len(kp),
            'num_keyframes': len(self.keyframes)
        }
    
    def _match_features(self, des1, des2):
        """Match features between frames"""
        if des1 is None or des2 is None:
            return []
        matches = self.matcher.match(des1, des2)
        return sorted(matches, key=lambda x: x.distance)
    
    def _estimate_pose(self, kp1, kp2, matches):
        """Estimate relative pose from matches"""
        if len(matches) < 8:
            return None
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        # Find essential matrix
        E, mask = cv2.findEssentialMat(
            pts1, pts2, self.K, method=cv2.RANSAC
        )
        
        if E is None:
            return None
        
        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, self.K)
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t.squeeze()
        
        return T
    
    def _is_keyframe(self, kp, des) -> bool:
        """Determine if current frame should be keyframe"""
        # Simple heuristic: every 10th frame or significant motion
        if len(self.keyframes) == 0:
            return True
        if len(self.keyframes) % 10 == 0:
            return True
        return False

class DistributedSLAM:
    def __init__(self, num_drones: int):
        """Initialize distributed SLAM for multiple drones"""
        self.slam_instances = {}
        self.global_map = GlobalMap()
        self.map_lock = threading.Lock()
        
    def add_drone(self, drone_id: str, camera_params: Dict):
        """Add new drone to SLAM system"""
        K = camera_params['camera_matrix']
        dist = camera_params['dist_coeffs']
        self.slam_instances[drone_id] = VisualSLAM(K, dist)
    
    def process_frame(self, drone_id: str, frame: np.ndarray, 
                     timestamp: float) -> Dict:
        """Process frame from specific drone"""
        if drone_id not in self.slam_instances:
            return {}
        
        # Local SLAM processing
        local_result = self.slam_instances[drone_id].process_frame(
            frame, timestamp
        )
        
        # Update global map
        with self.map_lock:
            self.global_map.integrate_local_map(
                drone_id, local_result
            )
        
        return local_result
    
    def get_global_map(self) -> Dict:
        """Get current global map"""
        with self.map_lock:
            return self.global_map.get_map()

class GlobalMap:
    def __init__(self):
        """Initialize global map fusion"""
        self.drone_poses = {}
        self.global_points = []
        
    def integrate_local_map(self, drone_id: str, local_data: Dict):
        """Integrate local SLAM data into global map"""
        self.drone_poses[drone_id] = local_data.get('pose', np.eye(4))
        
        # TODO: Implement map merging logic
        # - Transform local points to global frame
        # - Detect loop closures between drones
        # - Optimize global map
    
    def get_map(self) -> Dict:
        """Return current global map state"""
        return {
            'drone_poses': self.drone_poses.copy(),
            'num_points': len(self.global_points)
        }
```

## 5. Betaflight Control Interface

### 5.1 MSP Protocol Implementation

```python
# betaflight_controller.py
import serial
import struct
import time
from enum import IntEnum
from typing import List, Dict, Optional

class MSPCodes(IntEnum):
    """MSP command codes"""
    MSP_STATUS = 101
    MSP_RAW_IMU = 102
    MSP_RC = 105
    MSP_ATTITUDE = 108
    MSP_ALTITUDE = 109
    MSP_SET_RAW_RC = 200
    MSP_SET_PID = 202

class BetaflightController:
    def __init__(self, port: str, baudrate: int = 115200):
        """
        Initialize Betaflight controller
        Args:
            port: Serial port (e.g., '/dev/ttyUSB0' or 'COM3')
            baudrate: Serial baudrate
        """
        self.serial = serial.Serial(port, baudrate, timeout=0.1)
        self.last_rc_time = 0
        self.min_rc_interval = 0.02  # 50Hz max update rate
    
    def _send_msp(self, code: int, data: bytes = b''):
        """Send MSP command"""
        # MSP header: $M<
        header = b'$M<'
        size = len(data)
        
        # Build message
        msg = header + bytes([size, code])
        msg += data
        
        # Calculate checksum
        checksum = size ^ code
        for byte in data:
            checksum ^= byte
        msg += bytes([checksum])
        
        self.serial.write(msg)
    
    def _receive_msp(self) -> Optional[bytes]:
        """Receive MSP response"""
        # Wait for header
        header = self.serial.read(3)
        if header != b'$M>':
            return None
        
        # Read size and code
        size = ord(self.serial.read(1))
        code = ord(self.serial.read(1))
        
        # Read data
        data = self.serial.read(size)
        
        # Read and verify checksum
        checksum = ord(self.serial.read(1))
        calc_checksum = size ^ code
        for byte in data:
            calc_checksum ^= byte
        
        if checksum != calc_checksum:
            return None
        
        return data
    
    def send_rc_commands(self, roll: int, pitch: int, 
                         yaw: int, throttle: int,
                         aux1: int = 1000, aux2: int = 1000):
        """
        Send RC commands to drone
        Args:
            roll, pitch, yaw, throttle: Values 1000-2000
            aux1, aux2: Auxiliary channels
        """
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_rc_time < self.min_rc_interval:
            return
        
        # Pack RC channels (16 channels, 2 bytes each)
        channels = [roll, pitch, throttle, yaw, 
                   aux1, aux2, 1500, 1500,
                   1500, 1500, 1500, 1500,
                   1500, 1500, 1500, 1500]
        
        data = struct.pack('<' + 'H' * 16, *channels)
        self._send_msp(MSPCodes.MSP_SET_RAW_RC, data)
        self.last_rc_time = current_time
    
    def get_attitude(self) -> Dict:
        """Get current attitude (roll, pitch, yaw)"""
        self._send_msp(MSPCodes.MSP_ATTITUDE)
        data = self._receive_msp()
        
        if data and len(data) >= 6:
            roll, pitch, yaw = struct.unpack('<hhh', data[:6])
            return {
                'roll': roll / 10.0,  # Convert to degrees
                'pitch': pitch / 10.0,
                'yaw': yaw
            }
        return {}
    
    def get_altitude(self) -> float:
        """Get current altitude in meters"""
        self._send_msp(MSPCodes.MSP_ALTITUDE)
        data = self._receive_msp()
        
        if data and len(data) >= 6:
            alt_cm, vario = struct.unpack('<ih', data[:6])
            return alt_cm / 100.0  # Convert to meters
        return 0.0
    
    def arm(self):
        """Arm the drone"""
        # Arm sequence: throttle low, yaw right
        for _ in range(20):
            self.send_rc_commands(1500, 1500, 2000, 1000)
            time.sleep(0.1)
    
    def disarm(self):
        """Disarm the drone"""
        # Disarm sequence: throttle low, yaw left
        for _ in range(20):
            self.send_rc_commands(1500, 1500, 1000, 1000)
            time.sleep(0.1)

class SwarmController:
    def __init__(self, drone_configs: Dict[str, str]):
        """
        Initialize swarm controller
        Args:
            drone_configs: {drone_id: serial_port}
        """
        self.controllers = {}
        for drone_id, port in drone_configs.items():
            self.controllers[drone_id] = BetaflightController(port)
    
    def send_formation_commands(self, formation: Dict[str, Dict]):
        """Send coordinated commands to maintain formation"""
        for drone_id, commands in formation.items():
            if drone_id in self.controllers:
                self.controllers[drone_id].send_rc_commands(
                    **commands
                )
    
    def emergency_stop_all(self):
        """Emergency stop all drones"""
        for controller in self.controllers.values():
            controller.disarm()
```

## 6. Main Application

### 6.1 Integrated System

```python
# main.py
import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List
import yaml

@dataclass
class SystemConfig:
    num_drones: int
    video_devices: List[int]
    serial_ports: List[str]
    model_path: str
    output_dir: str

class DroneSwarmSystem:
    def __init__(self, config_path: str):
        """Initialize complete drone swarm system"""
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.video_receiver = None
        self.detector = None
        self.slam = None
        self.controller = None
        
        self._setup_logging()
        self._initialize_components()
    
    def _load_config(self, path: str) -> SystemConfig:
        """Load system configuration"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return SystemConfig(**data)
    
    def _setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('drone_swarm.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all system components"""
        # Video reception
        drone_configs = [
            DroneStream(f"drone_{i}", self.config.video_devices[i])
            for i in range(self.config.num_drones)
        ]
        self.video_receiver = MultiStreamReceiver(drone_configs)
        
        # Object detection
        self.detector = DroneTargetDetector(self.config.model_path)
        
        # SLAM
        self.slam = DistributedSLAM(self.config.num_drones)
        
        # Control
        drone_controls = {
            f"drone_{i}": self.config.serial_ports[i]
            for i in range(self.config.num_drones)
        }
        self.controller = SwarmController(drone_controls)
        
        self.logger.info("All components initialized")
    
    async def run(self):
        """Main processing loop"""
        self.logger.info("Starting drone swarm system")
        
        try:
            while True:
                # Get frames from all drones
                frames = self.video_receiver.get_all_frames()
                
                if frames:
                    # Run detection
                    detections = self.detector.detect_batch(frames)
                    
                    # Process SLAM for each drone
                    for drone_id, frame in frames.items():
                        slam_result = self.slam.process_frame(
                            drone_id, frame, time.time()
                        )
                    
                    # Compute control commands based on detections
                    commands = self._compute_control_commands(
                        detections, self.slam.get_global_map()
                    )
                    
                    # Send commands to drones
                    self.controller.send_formation_commands(commands)
                    
                    # Log status
                    self._log_status(detections, slam_result)
                
                await asyncio.sleep(0.033)  # ~30 FPS
                
        except KeyboardInterrupt:
            self.logger.info("Shutting down...")
            self.shutdown()
    
    def _compute_control_commands(self, detections: Dict, 
                                  global_map: Dict) -> Dict:
        """Compute control commands from detections and map"""
        commands = {}
        
        for drone_id in detections.keys():
            # Simple reactive control based on detections
            targets = detections[drone_id]
            
            if targets:
                # Track first target
                target = targets[0]
                bbox = target['bbox']
                
                # Compute centering commands
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                
                # Proportional control
                roll = int(1500 + (center_x - 960) * 0.5)
                pitch = int(1500 + (center_y - 540) * 0.5)
                
                commands[drone_id] = {
                    'roll': max(1000, min(2000, roll)),
                    'pitch': max(1000, min(2000, pitch)),
                    'yaw': 1500,
                    'throttle': 1500
                }
            else:
                # Hover in place
                commands[drone_id] = {
                    'roll': 1500,
                    'pitch': 1500,
                    'yaw': 1500,
                    'throttle': 1500
                }
        
        return commands
    
    def _log_status(self, detections: Dict, slam_result: Dict):
        """Log system status"""
        total_detections = sum(len(d) for d in detections.values())
        self.logger.info(
            f"Detections: {total_detections}, "
            f"SLAM keyframes: {slam_result.get('num_keyframes', 0)}"
        )
    
    def shutdown(self):
        """Clean shutdown"""
        self.controller.emergency_stop_all()
        self.logger.info("System shutdown complete")

if __name__ == "__main__":
    # Create config file
    config = {
        'num_drones': 3,
        'video_devices': [0, 1, 2],
        'serial_ports': ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2'],
        'model_path': 'yolov8n.pt',
        'output_dir': './output'
    }
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Run system
    system = DroneSwarmSystem('config.yaml')
    asyncio.run(system.run())
```

## 7. Testing & Simulation

### 7.1 Simulator Integration

```python
# simulator.py
import numpy as np
import cv2
from typing import List, Tuple

class DroneSimulator:
    """Simple drone swarm simulator for testing"""
    
    def __init__(self, num_drones: int, world_size: Tuple[int, int]):
        self.num_drones = num_drones
        self.world_size = world_size
        
        # Initialize drone positions randomly
        self.positions = np.random.rand(num_drones, 2) * world_size
        self.velocities = np.zeros((num_drones, 2))
        
        # Create synthetic environment
        self.environment = self._create_environment()
    
    def _create_environment(self) -> np.ndarray:
        """Create synthetic environment with targets"""
        env = np.ones((self.world_size[1], self.world_size[0], 3), 
                      dtype=np.uint8) * 255
        
        # Add some random targets
        for _ in range(10):
            x = np.random.randint(0, self.world_size[0])
            y = np.random.randint(0, self.world_size[1])
            cv2.circle(env, (x, y), 20, (0, 0, 255), -1)
        
        return env
    
    def step(self, commands: Dict[int, np.ndarray]):
        """Update simulation state"""
        for drone_id, command in commands.items():
            # Simple physics update
            self.velocities[drone_id] = command
            self.positions[drone_id] += self.velocities[drone_id]
            
            # Keep in bounds
            self.positions[drone_id] = np.clip(
                self.positions[drone_id], 
                [0, 0], self.world_size
            )
    
    def get_drone_view(self, drone_id: int, 
                       fov: int = 90) -> np.ndarray:
        """Get camera view from drone perspective"""
        # Extract region around drone
        x, y = self.positions[drone_id].astype(int)
        size = 320  # Camera resolution
        
        # Extract view (simplified)
        x1 = max(0, x - size // 2)
        x2 = min(self.world_size[0], x + size // 2)
        y1 = max(0, y - size // 2)
        y2 = min(self.world_size[1], y + size // 2)
        
        view = self.environment[y1:y2, x1:x2].copy()
        
        # Resize to standard size
        view = cv2.resize(view, (size, size))
        
        # Add drone overlay
        cv2.putText(view, f"Drone {drone_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return view
    
    def visualize(self) -> np.ndarray:
        """Create visualization of entire swarm"""
        vis = self.environment.copy()
        
        # Draw all drones
        for i, pos in enumerate(self.positions):
            x, y = pos.astype(int)
            cv2.circle(vis, (x, y), 10, (0, 255, 0), -1)
            cv2.putText(vis, str(i), (x-5, y+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return vis

# Test the simulator
if __name__ == "__main__":
    sim = DroneSimulator(num_drones=3, world_size=(800, 600))
    
    cv2.namedWindow("Swarm Simulation")
    
    while True:
        # Random commands for testing
        commands = {
            i: np.random.randn(2) * 2
            for i in range(sim.num_drones)
        }
        
        sim.step(commands)
        
        # Visualize
        vis = sim.visualize()
        cv2.imshow("Swarm Simulation", vis)
        
        # Show individual drone views
        for i in range(sim.num_drones):
            view = sim.get_drone_view(i)
            cv2.imshow(f"Drone {i} View", view)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
```

## Deployment Checklist

### Hardware Setup
- [ ] USB video capture devices connected and recognized
- [ ] Serial connections to Betaflight boards established
- [ ] GPU drivers installed (if using CUDA)
- [ ] 5.8GHz receivers tuned to correct channels

### Software Setup
- [ ] Python environment created and packages installed
- [ ] YOLO model downloaded and tested
- [ ] Camera calibration completed
- [ ] Serial port permissions configured (Linux: add user to dialout group)

### Safety Checks
- [ ] Emergency stop procedures tested
- [ ] Failsafe configurations verified
- [ ] GPS return-to-home backup (if available)
- [ ] Battery monitoring active
- [ ] Geofencing boundaries set

### Performance Optimization
- [ ] GPU acceleration enabled
- [ ] Video codec optimization
- [ ] Network latency minimized
- [ ] CPU affinity set for critical processes

## Troubleshooting

### Common Issues

1. **Video Latency**: Use hardware encoding/decoding when possible
2. **Detection Performance**: Reduce model size or input resolution
3. **Serial Communication**: Check baud rates and USB permissions
4. **SLAM Drift**: Increase keyframe frequency and feature count

## Next Steps

1. Implement advanced swarm behaviors (formation flight, search patterns)
2. Add mesh networking between drones
3. Integrate thermal cameras for night operations
4. Develop custom YOLO model trained on aerial footage
5. Implement predictive targeting algorithms
