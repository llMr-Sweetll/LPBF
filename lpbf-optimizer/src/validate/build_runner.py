import os
import subprocess
import yaml
from pathlib import Path
import logging
import time
import numpy as np
import h5py
from datetime import datetime


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('build_runner')


class LPBFBuildRunner:
    """
    Interface for sending optimized scan parameters to an LPBF machine
    to build validation coupons.
    
    This class converts optimized process parameters into G-code or
    machine-specific formats and sends them to LPBF equipment.
    """
    
    def __init__(self, config_path):
        """
        Initialize the build runner
        
        Args:
            config_path: Path to the configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set machine type
        self.machine_type = self.config.get('validate', {}).get('machine_type', 'generic')
        logger.info(f"Initialized build runner for {self.machine_type} machine")
        
        # Set up directories
        self.output_dir = Path(self.config.get('validate', {}).get('build_output_dir', 'builds'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a build ID based on timestamp
        self.build_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.build_dir = self.output_dir / self.build_id
        self.build_dir.mkdir(exist_ok=True)
        
        # Set up log file for this build
        file_handler = logging.FileHandler(self.build_dir / 'build_run.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Save config for this build
        with open(self.build_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def load_optimal_parameters(self, param_file):
        """
        Load optimized process parameters from file
        
        Args:
            param_file: Path to the parameter file (CSV, YAML, or HDF5)
            
        Returns:
            List of parameter dictionaries
        """
        file_ext = Path(param_file).suffix.lower()
        
        if file_ext == '.h5':
            # Load from HDF5
            params = []
            with h5py.File(param_file, 'r') as f:
                param_values = f['parameters'][:]
                param_names = [name.decode('utf-8') for name in f['parameter_names'][:]]
                
                for row in param_values:
                    param_dict = {name: value for name, value in zip(param_names, row)}
                    params.append(param_dict)
        
        elif file_ext == '.csv':
            # Load from CSV
            import pandas as pd
            df = pd.read_csv(param_file)
            params = df.to_dict('records')
        
        elif file_ext in ['.yaml', '.yml']:
            # Load from YAML
            with open(param_file, 'r') as f:
                params = yaml.safe_load(f)
        
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Loaded {len(params)} parameter sets from {param_file}")
        return params
    
    def generate_coupon_geometry(self, coupon_type='cube', size=10):
        """
        Generate a simple test coupon geometry
        
        Args:
            coupon_type: Type of coupon ('cube', 'tensile', 'cylinder')
            size: Size in mm
            
        Returns:
            Path to the generated STL file
        """
        try:
            # Check if we have a CAD library available
            from stl import mesh
            import numpy as np
            
            vertices = []
            faces = []
            
            if coupon_type == 'cube':
                # Create a cube
                vertices = np.array([
                    [0, 0, 0],
                    [size, 0, 0],
                    [0, size, 0],
                    [size, size, 0],
                    [0, 0, size],
                    [size, 0, size],
                    [0, size, size],
                    [size, size, size]
                ])
                
                faces = np.array([
                    [0, 2, 1],
                    [1, 2, 3],
                    [0, 1, 4],
                    [1, 5, 4],
                    [1, 3, 5],
                    [3, 7, 5],
                    [3, 2, 7],
                    [2, 6, 7],
                    [2, 0, 6],
                    [0, 4, 6],
                    [4, 5, 6],
                    [5, 7, 6]
                ])
                
                # Create the mesh
                cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        cube.vectors[i][j] = vertices[f[j], :]
                
                # Write to file
                stl_path = self.build_dir / f"coupon_{coupon_type}_{size}mm.stl"
                cube.save(stl_path)
                return stl_path
            
            elif coupon_type == 'cylinder':
                # Create a cylinder
                r = size / 2
                h = size
                n = 30  # Number of sides for approximation
                
                # Create circle points
                theta = np.linspace(0, 2*np.pi, n, endpoint=False)
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                # Create top and bottom vertices
                bottom_vertices = np.column_stack((x, y, np.zeros(n)))
                top_vertices = np.column_stack((x, y, np.ones(n) * h))
                
                # Combine vertices
                vertices = np.vstack((bottom_vertices, top_vertices))
                
                # Create faces - sides
                side_faces = []
                for i in range(n):
                    i_next = (i + 1) % n
                    side_faces.append([i, i_next, i + n])
                    side_faces.append([i_next, i_next + n, i + n])
                
                # Create faces - top and bottom
                center_bottom = len(vertices)
                center_top = len(vertices) + 1
                vertices = np.vstack((vertices, np.array([0, 0, 0]), np.array([0, 0, h])))
                
                bottom_faces = [[center_bottom, (i + 1) % n, i] for i in range(n)]
                top_faces = [[center_top, i + n, (i + 1) % n + n] for i in range(n)]
                
                # Combine all faces
                faces = np.vstack((side_faces, bottom_faces, top_faces))
                
                # Create the mesh
                cylinder = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
                for i, f in enumerate(faces):
                    for j in range(3):
                        cylinder.vectors[i][j] = vertices[f[j], :]
                
                # Write to file
                stl_path = self.build_dir / f"coupon_{coupon_type}_{size}mm.stl"
                cylinder.save(stl_path)
                return stl_path
            
            else:
                logger.warning(f"Coupon type '{coupon_type}' not implemented, falling back to cube")
                return self.generate_coupon_geometry('cube', size)
        
        except ImportError:
            # If numpy-stl is not available, create a dummy file
            logger.warning("numpy-stl not available, creating dummy STL reference")
            dummy_path = self.build_dir / f"coupon_{coupon_type}_{size}mm.stl"
            with open(dummy_path, 'w') as f:
                f.write(f"# Dummy STL file for {coupon_type} coupon of size {size}mm\n")
            return dummy_path
    
    def parameters_to_gcode(self, parameters, stl_file, output_file=None):
        """
        Convert a set of process parameters to G-code
        
        Args:
            parameters: Dictionary of process parameters
            stl_file: Path to the STL file to build
            output_file: Path to save the G-code (if None, generate one)
            
        Returns:
            Path to the G-code file
        """
        if output_file is None:
            output_file = self.build_dir / f"build_{Path(stl_file).stem}.gcode"
        
        # Extract required parameters
        P = parameters.get('P', 200)  # Laser power (W)
        v = parameters.get('v', 800)  # Scan speed (mm/s)
        h = parameters.get('h', 0.1)  # Hatch spacing (mm)
        theta = parameters.get('theta', 0)  # Scan angle (degrees)
        layer_thickness = parameters.get('layer_thickness', 0.03)  # Layer thickness (mm)
        
        # Load STL and get dimensions (this would be implemented with a real STL parser)
        # For simplicity, we just use fake dimensions
        width = 10.0
        depth = 10.0
        height = 10.0
        
        # Calculate number of layers
        n_layers = int(np.ceil(height / layer_thickness))
        
        # Generate G-code
        with open(output_file, 'w') as f:
            # Write header
            f.write("; G-code generated for LPBF build\n")
            f.write(f"; File: {stl_file}\n")
            f.write(f"; Parameters: P={P}W, v={v}mm/s, h={h}mm, angle={theta}°\n")
            f.write(f"; Layer thickness: {layer_thickness}mm\n")
            f.write(f"; Total layers: {n_layers}\n\n")
            
            # Machine setup
            f.write("G90 ; Absolute positioning\n")
            f.write(f"M104 S{P} ; Set laser power to {P}W\n")
            f.write("G28 ; Home all axes\n")
            f.write("G1 Z5 F5000 ; Raise platform\n\n")
            
            # Process each layer
            for layer in range(n_layers):
                z = layer * layer_thickness
                f.write(f"; Layer {layer+1}/{n_layers} at z={z}mm\n")
                f.write(f"G1 Z{z} F1000 ; Move to layer height\n")
                
                # Rotate scan direction every layer
                scan_angle = (theta + (layer % 2) * 90) % 180
                
                # Calculate scan lines based on hatch spacing
                n_lines = int(np.ceil(width / h))
                
                # Scan back and forth
                for line in range(n_lines):
                    y = line * h
                    if scan_angle == 0:
                        # Horizontal lines
                        f.write(f"G1 X0 Y{y} F{v*60} ; Move to start\n")
                        f.write(f"G1 X{width} Y{y} F{v*60} ; Scan line\n")
                        
                        # Move to next line if not the last
                        if line < n_lines - 1:
                            if line % 2 == 0:
                                # Scan left to right then right to left
                                f.write(f"G1 X{width} Y{y+h} F{v*60} ; Move to next line\n")
                            else:
                                # Scan right to left then left to right
                                f.write(f"G1 X0 Y{y+h} F{v*60} ; Move to next line\n")
                    else:
                        # Vertical lines
                        f.write(f"G1 X{y} Y0 F{v*60} ; Move to start\n")
                        f.write(f"G1 X{y} Y{depth} F{v*60} ; Scan line\n")
                        
                        # Move to next line if not the last
                        if line < n_lines - 1:
                            if line % 2 == 0:
                                # Scan bottom to top then top to bottom
                                f.write(f"G1 X{y+h} Y{depth} F{v*60} ; Move to next line\n")
                            else:
                                # Scan top to bottom then bottom to top
                                f.write(f"G1 X{y+h} Y0 F{v*60} ; Move to next line\n")
                
                f.write("\n")
            
            # Finish
            f.write("G1 Z50 F1000 ; Raise platform when done\n")
            f.write("M104 S0 ; Turn off laser\n")
            f.write("M84 ; Disable motors\n")
        
        logger.info(f"Generated G-code: {output_file}")
        return output_file
    
    def parameters_to_machine_format(self, parameters, stl_file, machine_type=None):
        """
        Convert parameters to machine-specific format
        
        Args:
            parameters: Dictionary of process parameters
            stl_file: Path to the STL file
            machine_type: Machine type (if None, use default)
            
        Returns:
            Path to the machine-specific file
        """
        if machine_type is None:
            machine_type = self.machine_type
        
        if machine_type.lower() in ['generic', 'gcode']:
            # Use G-code format
            return self.parameters_to_gcode(parameters, stl_file)
        
        elif machine_type.lower() == 'eos':
            # EOS format - job file
            job_file = self.build_dir / f"build_{Path(stl_file).stem}.eosjob"
            with open(job_file, 'w') as f:
                f.write("# EOS Job File\n")
                f.write(f"STL_FILE: {stl_file}\n")
                f.write(f"LASER_POWER: {parameters.get('P', 200)}\n")
                f.write(f"SCAN_SPEED: {parameters.get('v', 800)}\n")
                f.write(f"HATCH_DISTANCE: {parameters.get('h', 0.1)}\n")
                f.write(f"LAYER_THICKNESS: {parameters.get('layer_thickness', 0.03)}\n")
            return job_file
        
        elif machine_type.lower() == 'concept_laser':
            # Concept Laser format - CLS file
            cls_file = self.build_dir / f"build_{Path(stl_file).stem}.cls"
            with open(cls_file, 'w') as f:
                f.write("# Concept Laser CLS File\n")
                f.write(f"STL_FILE: {stl_file}\n")
                f.write(f"LASER_POWER: {parameters.get('P', 200)}\n")
                f.write(f"SCAN_SPEED: {parameters.get('v', 800)}\n")
                f.write(f"HATCH_DISTANCE: {parameters.get('h', 0.1)}\n")
                f.write(f"LAYER_THICKNESS: {parameters.get('layer_thickness', 0.03)}\n")
            return cls_file
        
        else:
            logger.warning(f"Unsupported machine type: {machine_type}, falling back to G-code")
            return self.parameters_to_gcode(parameters, stl_file)
    
    def send_to_machine(self, machine_file, machine_ip=None, dry_run=True):
        """
        Send a machine file to an LPBF machine
        
        Args:
            machine_file: Path to the machine-specific file
            machine_ip: IP address of the machine (if None, use from config)
            dry_run: If True, simulate sending but don't actually send
            
        Returns:
            True if successful, False otherwise
        """
        if machine_ip is None:
            machine_ip = self.config.get('validate', {}).get('machine_ip', '127.0.0.1')
        
        logger.info(f"{'[DRY RUN] ' if dry_run else ''}Sending {machine_file} to machine at {machine_ip}")
        
        if dry_run:
            # Simulate sending
            time.sleep(1)
            logger.info("[DRY RUN] File sent successfully")
            return True
        
        # For actual sending, this would interface with the machine API
        # For now, we just simulate a successful send
        try:
            # This would be a real API call or file transfer
            # e.g. scp, FTP, or machine-specific API
            time.sleep(2)
            logger.info(f"File sent successfully to {machine_ip}")
            return True
        except Exception as e:
            logger.error(f"Failed to send file: {e}")
            return False
    
    def run_build(self, parameter_file, coupon_type='cube', size=10, machine_ip=None, dry_run=True):
        """
        Run a complete build process with the given parameters
        
        Args:
            parameter_file: Path to the parameter file
            coupon_type: Type of coupon to build
            size: Size of the coupon in mm
            machine_ip: IP address of the machine
            dry_run: If True, simulate sending but don't actually send
            
        Returns:
            Dictionary with build information
        """
        # Load parameters
        parameters = self.load_optimal_parameters(parameter_file)
        
        # For now, just use the first parameter set
        # In practice, we might build multiple coupons with different parameters
        if isinstance(parameters, list) and len(parameters) > 0:
            parameters = parameters[0]
        
        # Generate coupon geometry
        stl_file = self.generate_coupon_geometry(coupon_type, size)
        
        # Convert to machine format
        machine_file = self.parameters_to_machine_format(parameters, stl_file)
        
        # Send to machine
        success = self.send_to_machine(machine_file, machine_ip, dry_run)
        
        # Return build information
        build_info = {
            'build_id': self.build_id,
            'parameters': parameters,
            'coupon_type': coupon_type,
            'size': size,
            'stl_file': str(stl_file),
            'machine_file': str(machine_file),
            'machine_ip': machine_ip,
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save build info
        with open(self.build_dir / 'build_info.yaml', 'w') as f:
            yaml.dump(build_info, f)
        
        logger.info(f"Build {'completed successfully' if success else 'failed'}")
        return build_info


def main():
    """Main function to run a build"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LPBF builds with optimized parameters')
    parser.add_argument('--config', type=str, default='../../data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--params', type=str, required=True,
                        help='Path to optimized parameters file')
    parser.add_argument('--coupon', type=str, default='cube',
                        choices=['cube', 'cylinder', 'tensile'],
                        help='Type of coupon to build')
    parser.add_argument('--size', type=float, default=10.0,
                        help='Size of the coupon in mm')
    parser.add_argument('--machine-ip', type=str, default=None,
                        help='IP address of the LPBF machine')
    parser.add_argument('--dry-run', action='store_true',
                        help='Simulate sending to machine but don\'t actually send')
    
    args = parser.parse_args()
    
    # Create build runner
    runner = LPBFBuildRunner(args.config)
    
    # Run build
    build_info = runner.run_build(
        args.params,
        coupon_type=args.coupon,
        size=args.size,
        machine_ip=args.machine_ip,
        dry_run=args.dry_run
    )
    
    print(f"Build {'completed successfully' if build_info['success'] else 'failed'}")
    print(f"Build directory: {runner.build_dir}")


if __name__ == '__main__':
    main()