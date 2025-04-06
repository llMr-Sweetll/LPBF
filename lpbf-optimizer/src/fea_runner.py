import os
import subprocess
import tempfile
import time
import yaml
import h5py
import numpy as np
from pathlib import Path
import logging
from datetime import datetime


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('fea_runner')


class FEARunner:
    """
    Wrapper for running Finite Element Analysis (FEA) simulations
    for LPBF process with commercial solvers like ABAQUS or COMSOL.
    """
    
    def __init__(self, config_path):
        """
        Initialize the FEA runner
        
        Args:
            config_path: Path to the configuration file with FEA settings
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set solver
        self.solver_type = self.config['fea']['solver_type']
        logger.info(f"Initialized FEA runner with {self.solver_type} solver")
        
        # Set up output paths
        self.output_dir = Path(self.config['fea']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a run ID based on timestamp
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        # Set up log file for this run
        file_handler = logging.FileHandler(self.run_dir / 'fea_run.log')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        
        # Save config for this run
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def generate_input_file(self, scan_vector, template_path=None):
        """
        Generate solver input file from template with scan vector parameters
        
        Args:
            scan_vector: Dictionary of scan parameters (P, v, h, etc.)
            template_path: Path to the template file (if None, use default from config)
            
        Returns:
            Path to the generated input file
        """
        if template_path is None:
            template_path = self.config['fea']['template_path']
        
        # Load template
        with open(template_path, 'r') as f:
            template = f.read()
        
        # Replace placeholders with scan vector parameters
        for key, value in scan_vector.items():
            placeholder = f'${{{key}}}'
            template = template.replace(placeholder, str(value))
        
        # Write to a temporary file
        input_file = self.run_dir / f"input_{int(time.time())}.inp"
        with open(input_file, 'w') as f:
            f.write(template)
        
        logger.info(f"Generated input file: {input_file}")
        return input_file
    
    def run_abaqus_job(self, input_file, job_name=None):
        """
        Run an ABAQUS simulation job
        
        Args:
            input_file: Path to the ABAQUS input file
            job_name: Name for the job (if None, use basename of input file)
            
        Returns:
            Path to output database (.odb) file
        """
        if job_name is None:
            job_name = Path(input_file).stem
        
        # Build command
        abaqus_cmd = [
            self.config['fea']['abaqus_path'],
            'job=%s' % job_name,
            'input=%s' % input_file,
            'interactive',
            'cpus=%d' % self.config['fea'].get('n_cpus', 4)
        ]
        
        # Log command
        logger.info(f"Running ABAQUS command: {' '.join(abaqus_cmd)}")
        
        # Execute
        try:
            subprocess.run(
                abaqus_cmd, 
                check=True,
                cwd=self.run_dir
            )
            logger.info(f"ABAQUS job {job_name} completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"ABAQUS job failed with error: {e}")
            raise
        
        # Return path to output database
        odb_file = self.run_dir / f"{job_name}.odb"
        return odb_file
    
    def run_comsol_job(self, input_file, job_name=None):
        """
        Run a COMSOL simulation job
        
        Args:
            input_file: Path to the COMSOL input file (.mph or .java)
            job_name: Name for the job (if None, use basename of input file)
            
        Returns:
            Path to output file
        """
        if job_name is None:
            job_name = Path(input_file).stem
        
        # Build command
        comsol_cmd = [
            self.config['fea']['comsol_path'],
            'batch',
            '-inputfile', str(input_file),
            '-outputfile', f"{job_name}_out.mph"
        ]
        
        # Log command
        logger.info(f"Running COMSOL command: {' '.join(comsol_cmd)}")
        
        # Execute
        try:
            subprocess.run(
                comsol_cmd, 
                check=True,
                cwd=self.run_dir
            )
            logger.info(f"COMSOL job {job_name} completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"COMSOL job failed with error: {e}")
            raise
        
        # Return path to output file
        out_file = self.run_dir / f"{job_name}_out.mph"
        return out_file
    
    def extract_abaqus_results(self, odb_file, output_file=None):
        """
        Extract results from ABAQUS .odb file using Python script
        
        Args:
            odb_file: Path to ABAQUS .odb file
            output_file: Path to output HDF5 file (if None, generate one)
            
        Returns:
            Path to HDF5 file with extracted results
        """
        if output_file is None:
            output_file = self.run_dir / f"{Path(odb_file).stem}_results.h5"
        
        # Create a temporary Python script to extract data using Abaqus Python API
        extract_script = self.run_dir / "extract_results.py"
        
        with open(extract_script, 'w') as f:
            f.write("""
from odbAccess import *
from abaqusConstants import *
import numpy as np
import h5py
import sys

def extract_to_hdf5(odb_path, output_path):
    # Open the ODB file
    odb = openOdb(path=odb_path, readOnly=True)
    
    # Get the last frame of the last step
    step = odb.steps[odb.steps.keys()[-1]]
    last_frame = step.frames[-1]
    
    # Extract field outputs
    stress = last_frame.fieldOutputs['S']
    temperature = last_frame.fieldOutputs['NT11']
    
    # Get node and element data
    nodeCoords = {}
    for node in odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]].nodes:
        nodeCoords[node.label] = node.coordinates
    
    # Create numpy arrays
    node_ids = []
    coordinates = []
    von_mises_stress = []
    temps = []
    
    # Extract stress and temperature at nodes
    for value in stress.values:
        node_id = value.nodeLabel
        if node_id in nodeCoords:
            node_ids.append(node_id)
            coordinates.append(nodeCoords[node_id])
            von_mises_stress.append(value.mises)
    
    for value in temperature.values:
        node_id = value.nodeLabel
        if node_id in nodeCoords:
            temps.append(value.data)
    
    # Save to HDF5
    with h5py.File(output_path, 'w') as f:
        # Create datasets
        f.create_dataset('node_ids', data=np.array(node_ids))
        f.create_dataset('coordinates', data=np.array(coordinates))
        f.create_dataset('von_mises_stress', data=np.array(von_mises_stress))
        f.create_dataset('temperature', data=np.array(temps))
        
        # Store metadata
        f.attrs['odb_file'] = odb_path
        f.attrs['num_nodes'] = len(node_ids)
    
    # Close the ODB file
    odb.close()
    
    print(f"Extracted data to {output_path}")

# Run extraction
extract_to_hdf5(r'{odb_file}', r'{output_file}')
""")
        
        # Build command to run the script with Abaqus Python
        abaqus_cmd = [
            self.config['fea']['abaqus_path'],
            'python',
            str(extract_script)
        ]
        
        # Log command
        logger.info(f"Running ABAQUS Python command: {' '.join(abaqus_cmd)}")
        
        # Execute
        try:
            subprocess.run(
                abaqus_cmd, 
                check=True,
                cwd=self.run_dir
            )
            logger.info(f"Data extracted to {output_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Data extraction failed with error: {e}")
            raise
        
        return output_file
    
    def extract_comsol_results(self, mph_file, output_file=None):
        """
        Extract results from COMSOL .mph file
        
        Args:
            mph_file: Path to COMSOL .mph file
            output_file: Path to output HDF5 file (if None, generate one)
            
        Returns:
            Path to HDF5 file with extracted results
        """
        if output_file is None:
            output_file = self.run_dir / f"{Path(mph_file).stem}_results.h5"
        
        # Create a Java script to extract data using COMSOL API
        extract_script = self.run_dir / "extract_results.java"
        
        with open(extract_script, 'w') as f:
            f.write(f"""
import com.comsol.model.*;
import com.comsol.model.util.*;

public class ExtractResults {{
    public static void main(String[] args) {{
        try {{
            ModelUtil.connect();
            
            // Load the model
            Model model = ModelUtil.load("{mph_file}");
            
            // Export data to VTK for easy access
            String vtk_file = "{self.run_dir}/result.vtk";
            model.result().export().create("vtk1", "VTK");
            model.result().export("vtk1").set("filename", vtk_file);
            model.result().export("vtk1").run();
            
            ModelUtil.disconnect();
            
            System.out.println("Exported results to " + vtk_file);
            System.exit(0);
        }} catch (Exception e) {{
            e.printStackTrace();
            System.exit(1);
        }}
    }}
}}
""")
        
        # Build command to compile and run the Java script
        javac_cmd = ["javac", "-cp", self.config['fea']['comsol_java_path'], str(extract_script)]
        java_cmd = [
            "java", 
            "-cp", f"{self.config['fea']['comsol_java_path']}:{self.run_dir}", 
            "ExtractResults"
        ]
        
        # Log commands
        logger.info(f"Compiling Java: {' '.join(javac_cmd)}")
        logger.info(f"Running Java: {' '.join(java_cmd)}")
        
        # Execute
        try:
            subprocess.run(javac_cmd, check=True, cwd=self.run_dir)
            subprocess.run(java_cmd, check=True, cwd=self.run_dir)
            
            # Convert VTK to HDF5 (this would be a separate helper function)
            self.vtk_to_hdf5(self.run_dir / "result.vtk", output_file)
            
            logger.info(f"Data extracted to {output_file}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Data extraction failed with error: {e}")
            raise
        
        return output_file
    
    def vtk_to_hdf5(self, vtk_file, hdf5_file):
        """
        Convert VTK file to HDF5 format
        
        Args:
            vtk_file: Path to VTK file
            hdf5_file: Path to output HDF5 file
        """
        # In a real implementation, we would use vtk/pyvista library to read the file
        # and convert to HDF5. Here we just provide a stub.
        logger.info(f"Converting {vtk_file} to {hdf5_file}")
        
        # Placeholder - in a real implementation, this would use the VTK Python API
        with h5py.File(hdf5_file, 'w') as f:
            f.attrs['vtk_file'] = str(vtk_file)
            f.attrs['conversion_time'] = datetime.now().isoformat()
            
            # Create dummy datasets for illustration
            f.create_dataset('coordinates', data=np.random.rand(1000, 3))
            f.create_dataset('von_mises_stress', data=np.random.rand(1000))
            f.create_dataset('temperature', data=np.random.rand(1000))
    
    def run_parameter_sweep(self, scan_vectors, parallel=False, max_workers=None):
        """
        Run a sweep of simulations with different parameter combinations
        
        Args:
            scan_vectors: List of dictionaries with scan parameters
            parallel: Whether to run simulations in parallel
            max_workers: Maximum number of parallel workers (if None, use all available)
            
        Returns:
            List of paths to result files
        """
        result_files = []
        
        if parallel:
            # Import here to avoid dependency if not used
            import concurrent.futures
            
            # Set up executor
            if max_workers is None:
                max_workers = self.config['fea'].get('max_parallel', 4)
            
            # Define worker function
            def run_simulation(scan_vector):
                job_name = f"job_{int(time.time())}_{scan_vector.get('id', 'sim')}"
                input_file = self.generate_input_file(scan_vector)
                
                if self.solver_type.lower() == 'abaqus':
                    odb_file = self.run_abaqus_job(input_file, job_name)
                    result_file = self.extract_abaqus_results(odb_file)
                elif self.solver_type.lower() == 'comsol':
                    mph_file = self.run_comsol_job(input_file, job_name)
                    result_file = self.extract_comsol_results(mph_file)
                else:
                    raise ValueError(f"Unsupported solver type: {self.solver_type}")
                
                # Add scan vector parameters to the result file
                self.append_scan_vector_to_h5(result_file, scan_vector)
                
                return result_file
            
            # Run in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(run_simulation, sv) for sv in scan_vectors]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result_file = future.result()
                        result_files.append(result_file)
                        logger.info(f"Completed simulation, result saved to {result_file}")
                    except Exception as e:
                        logger.error(f"Simulation failed with error: {e}")
        else:
            # Run sequentially
            for i, scan_vector in enumerate(scan_vectors):
                logger.info(f"Running simulation {i+1}/{len(scan_vectors)}")
                
                job_name = f"job_{int(time.time())}_{scan_vector.get('id', 'sim')}"
                input_file = self.generate_input_file(scan_vector)
                
                try:
                    if self.solver_type.lower() == 'abaqus':
                        odb_file = self.run_abaqus_job(input_file, job_name)
                        result_file = self.extract_abaqus_results(odb_file)
                    elif self.solver_type.lower() == 'comsol':
                        mph_file = self.run_comsol_job(input_file, job_name)
                        result_file = self.extract_comsol_results(mph_file)
                    else:
                        raise ValueError(f"Unsupported solver type: {self.solver_type}")
                    
                    # Add scan vector parameters to the result file
                    self.append_scan_vector_to_h5(result_file, scan_vector)
                    
                    result_files.append(result_file)
                    logger.info(f"Completed simulation, result saved to {result_file}")
                except Exception as e:
                    logger.error(f"Simulation {i+1} failed with error: {e}")
        
        return result_files
    
    def append_scan_vector_to_h5(self, h5_file, scan_vector):
        """
        Append scan vector parameters to HDF5 result file
        
        Args:
            h5_file: Path to HDF5 file
            scan_vector: Dictionary with scan parameters
        """
        with h5py.File(h5_file, 'a') as f:
            # Create a scan_vector group if it doesn't exist
            if 'scan_vector' not in f:
                sv_group = f.create_group('scan_vector')
            else:
                sv_group = f['scan_vector']
            
            # Add each parameter
            for key, value in scan_vector.items():
                sv_group.attrs[key] = value
    
    def combine_results(self, result_files, output_file=None):
        """
        Combine multiple simulation results into a single HDF5 file
        
        Args:
            result_files: List of paths to HDF5 result files
            output_file: Path to combined output file (if None, generate one)
            
        Returns:
            Path to combined result file
        """
        if output_file is None:
            output_file = self.run_dir / f"combined_results_{self.run_id}.h5"
        
        with h5py.File(output_file, 'w') as out_f:
            # Create groups
            sim_group = out_f.create_group('simulations')
            sv_group = out_f.create_group('scan_vectors')
            
            # Add each result file
            for i, file_path in enumerate(result_files):
                with h5py.File(file_path, 'r') as in_f:
                    # Create a group for this simulation
                    sim_i = sim_group.create_group(f'sim_{i:04d}')
                    
                    # Copy datasets
                    for key in in_f.keys():
                        if key != 'scan_vector':  # Handle scan vector separately
                            in_f.copy(key, sim_i)
                    
                    # Extract scan vector parameters
                    if 'scan_vector' in in_f:
                        sv_i = sv_group.create_group(f'sv_{i:04d}')
                        for key, value in in_f['scan_vector'].attrs.items():
                            sv_i.attrs[key] = value
            
            # Add metadata
            out_f.attrs['n_simulations'] = len(result_files)
            out_f.attrs['creation_time'] = datetime.now().isoformat()
            out_f.attrs['fea_solver'] = self.solver_type
        
        logger.info(f"Combined {len(result_files)} simulation results into {output_file}")
        return output_file


def main():
    """Main function to run FEA simulations"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FEA simulations for LPBF process')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--params', type=str, required=True,
                        help='Path to scan parameter file (YAML with list of parameter sets)')
    parser.add_argument('--parallel', action='store_true',
                        help='Run simulations in parallel')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: use config value)')
    
    args = parser.parse_args()
    
    # Create FEA runner
    runner = FEARunner(args.config)
    
    # Load scan vectors
    with open(args.params, 'r') as f:
        scan_vectors = yaml.safe_load(f)
    
    # Run parameter sweep
    result_files = runner.run_parameter_sweep(
        scan_vectors,
        parallel=args.parallel,
        max_workers=args.workers
    )
    
    # Combine results
    combined_file = runner.combine_results(result_files)
    
    print(f"FEA simulation completed. Combined results saved to {combined_file}")


if __name__ == '__main__':
    main()