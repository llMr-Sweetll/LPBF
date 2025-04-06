import os
import numpy as np
import h5py
import torch
from pathlib import Path
import yaml
import logging
from datetime import datetime
import argparse


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preprocessing')


class LPBFDataPreprocessor:
    """
    Preprocessor for LPBF FEA simulation data, converting raw outputs 
    (.odb, .vtk files) to tensors suitable for PINN training.
    """
    
    def __init__(self, config_path):
        """
        Initialize the preprocessor
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up paths
        self.raw_dir = Path(self.config['data']['raw_data_dir'])
        self.processed_path = Path(self.config['data']['processed_data_path'])
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check for existing processed file
        if self.processed_path.exists():
            logger.warning(f"Output file {self.processed_path} already exists. Will append new data.")
    
    def find_raw_files(self, pattern="*.h5"):
        """
        Find raw FEA simulation result files
        
        Args:
            pattern: File pattern to search for
            
        Returns:
            List of file paths
        """
        files = list(self.raw_dir.glob(pattern))
        logger.info(f"Found {len(files)} matching files")
        return files
    
    def load_fea_file(self, file_path):
        """
        Load a single FEA result file
        
        Args:
            file_path: Path to the HDF5 result file
            
        Returns:
            Dict containing the loaded data
        """
        logger.info(f"Loading {file_path}")
        
        data = {}
        
        with h5py.File(file_path, 'r') as f:
            # Load node coordinates
            data['coordinates'] = np.array(f['coordinates'])
            
            # Load stress field
            data['von_mises_stress'] = np.array(f['von_mises_stress'])
            
            # Load temperature field
            data['temperature'] = np.array(f['temperature'])
            
            # Load scan vector parameters
            data['scan_vector'] = {}
            if 'scan_vector' in f:
                sv_group = f['scan_vector']
                for key in sv_group.attrs:
                    data['scan_vector'][key] = sv_group.attrs[key]
        
        return data
    
    def calculate_derived_quantities(self, data):
        """
        Calculate derived quantities from the raw data
        
        Args:
            data: Dict containing the raw data
            
        Returns:
            Dict with added derived quantities
        """
        # Calculate thermal gradient
        # In practice, this would involve spatial derivatives of temperature
        # Here we'll use a placeholder calculation
        temp_array = data['temperature']
        coords = data['coordinates']
        
        # Dummy thermal gradient calculation
        # In reality, we would use finite differences or interpolation
        thermal_gradient = np.zeros_like(coords)
        
        # Calculate porosity estimate based on temperature and cooling rate
        # This is a simplified model - in reality would be more complex
        # Here we use temperature as a proxy - points that reached high temperature
        # but cooled too quickly may have porosity
        max_temp = np.max(temp_array)
        porosity = np.clip((temp_array - 1000) / (max_temp - 1000), 0, 1) * 0.1
        
        # Geometric accuracy - deviation from intended geometry
        # In practice, this would compare to CAD model
        # Here we use a placeholder
        geo_accuracy = np.ones_like(temp_array) * 0.95  # 95% accuracy
        
        # Add to data dictionary
        data['thermal_gradient'] = thermal_gradient
        data['porosity'] = porosity
        data['geometric_accuracy'] = geo_accuracy
        
        return data
    
    def normalize_data(self, data):
        """
        Normalize the data to suitable ranges for neural network training
        
        Args:
            data: Dict containing the data
            
        Returns:
            Dict with normalized data and normalization parameters
        """
        norm_params = {}
        
        # Normalize coordinates to [-1, 1]
        coords = data['coordinates']
        coord_min = np.min(coords, axis=0)
        coord_max = np.max(coords, axis=0)
        norm_coords = 2 * (coords - coord_min) / (coord_max - coord_min) - 1
        
        norm_params['coord_min'] = coord_min
        norm_params['coord_max'] = coord_max
        
        # Normalize stress
        stress = data['von_mises_stress']
        stress_max = np.max(stress)
        stress_min = np.min(stress)
        norm_stress = (stress - stress_min) / (stress_max - stress_min)
        
        norm_params['stress_min'] = stress_min
        norm_params['stress_max'] = stress_max
        
        # Normalize temperature
        temp = data['temperature']
        temp_max = np.max(temp)
        temp_min = np.min(temp)
        norm_temp = (temp - temp_min) / (temp_max - temp_min)
        
        norm_params['temp_min'] = temp_min
        norm_params['temp_max'] = temp_max
        
        # Replace with normalized data
        data['coordinates'] = norm_coords
        data['von_mises_stress'] = norm_stress
        data['temperature'] = norm_temp
        data['norm_params'] = norm_params
        
        return data
    
    def process_file(self, file_path):
        """
        Process a single FEA result file
        
        Args:
            file_path: Path to the HDF5 result file
            
        Returns:
            Dict containing processed data
        """
        # Load raw data
        data = self.load_fea_file(file_path)
        
        # Calculate derived quantities
        data = self.calculate_derived_quantities(data)
        
        # Normalize data
        data = self.normalize_data(data)
        
        return data
    
    def process_all_files(self):
        """
        Process all FEA result files and combine into a single dataset
        
        Returns:
            Path to the processed dataset
        """
        # Find all result files
        files = self.find_raw_files()
        
        # Initialize lists to store processed data
        all_coords = []
        all_stress = []
        all_temp = []
        all_porosity = []
        all_geo_accuracy = []
        all_scan_vectors = []
        all_times = []
        
        # Process each file
        for file_path in files:
            try:
                data = self.process_file(file_path)
                
                # Extract data
                all_coords.append(data['coordinates'])
                all_stress.append(data['von_mises_stress'])
                all_temp.append(data['temperature'])
                all_porosity.append(data['porosity'])
                all_geo_accuracy.append(data['geometric_accuracy'])
                
                # Create time values (dummy here - in reality would be from simulation)
                n_points = len(data['coordinates'])
                times = np.ones((n_points, 1))  # Final time state
                all_times.append(times)
                
                # Store scan vector
                all_scan_vectors.append(data['scan_vector'])
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        # Combine all data
        coords = np.concatenate(all_coords, axis=0)
        stress = np.concatenate(all_stress, axis=0)
        temp = np.concatenate(all_temp, axis=0)
        porosity = np.concatenate(all_porosity, axis=0)
        geo_accuracy = np.concatenate(all_geo_accuracy, axis=0)
        times = np.concatenate(all_times, axis=0)
        
        # Create tensors for PINN training
        # Format scan vectors as a tensor
        scan_keys = set()
        for sv in all_scan_vectors:
            scan_keys.update(sv.keys())
        
        scan_keys = sorted(list(scan_keys))
        n_scan_params = len(scan_keys)
        
        # Fill scan vector tensor
        n_samples = len(all_scan_vectors)
        scan_vectors = np.zeros((n_samples, n_scan_params))
        
        for i, sv in enumerate(all_scan_vectors):
            for j, key in enumerate(scan_keys):
                if key in sv:
                    scan_vectors[i, j] = sv[key]
        
        # Determine number of points per simulation
        points_per_sim = len(coords) // n_samples
        
        # Repeat scan vectors for each point in the simulation
        expanded_scan_vectors = np.repeat(scan_vectors, points_per_sim, axis=0)
        
        # Combine inputs and outputs for neural network
        # Inputs: scan parameters, coordinates, time
        # Outputs: stress, porosity, geometric accuracy
        
        inputs = np.hstack([expanded_scan_vectors, coords, times])
        outputs = np.column_stack([stress, porosity, geo_accuracy])
        
        # Split data into train/val/test sets
        n_total = len(inputs)
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        # Randomly shuffle indices
        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train+n_val]
        test_indices = indices[n_train+n_val:]
        
        # Create final datasets
        train_inputs = inputs[train_indices]
        train_outputs = outputs[train_indices]
        train_coords = coords[train_indices]
        train_times = times[train_indices]
        
        val_inputs = inputs[val_indices]
        val_outputs = outputs[val_indices]
        val_coords = coords[val_indices]
        val_times = times[val_indices]
        
        test_inputs = inputs[test_indices]
        test_outputs = outputs[test_indices]
        test_coords = coords[test_indices]
        test_times = times[test_indices]
        
        # Save to HDF5 file
        with h5py.File(self.processed_path, 'w') as f:
            # Create groups
            train_group = f.create_group('train')
            val_group = f.create_group('val')
            test_group = f.create_group('test')
            meta_group = f.create_group('metadata')
            
            # Save training data
            train_group.create_dataset('inputs', data=train_inputs)
            train_group.create_dataset('outputs', data=train_outputs)
            train_group.create_dataset('coordinates', data=train_coords)
            train_group.create_dataset('time', data=train_times)
            
            # Save validation data
            val_group.create_dataset('inputs', data=val_inputs)
            val_group.create_dataset('outputs', data=val_outputs)
            val_group.create_dataset('coordinates', data=val_coords)
            val_group.create_dataset('time', data=val_times)
            
            # Save test data
            test_group.create_dataset('inputs', data=test_inputs)
            test_group.create_dataset('outputs', data=test_outputs)
            test_group.create_dataset('coordinates', data=test_coords)
            test_group.create_dataset('time', data=test_times)
            
            # Extract scan vectors for each split
            train_sv = np.unique(expanded_scan_vectors[train_indices], axis=0)
            val_sv = np.unique(expanded_scan_vectors[val_indices], axis=0)
            test_sv = np.unique(expanded_scan_vectors[test_indices], axis=0)
            
            train_group.create_dataset('scan_vectors', data=train_sv)
            val_group.create_dataset('scan_vectors', data=val_sv)
            test_group.create_dataset('scan_vectors', data=test_sv)
            
            # Save metadata
            meta_group.attrs['n_total'] = n_total
            meta_group.attrs['n_train'] = len(train_indices)
            meta_group.attrs['n_val'] = len(val_indices)
            meta_group.attrs['n_test'] = len(test_indices)
            meta_group.attrs['created'] = datetime.now().isoformat()
            
            # Save scan parameter names
            dt = h5py.special_dtype(vlen=str)
            param_names = np.array(scan_keys, dtype=object)
            meta_group.create_dataset('parameter_names', data=param_names, dtype=dt)
            
        logger.info(f"Processed dataset saved to {self.processed_path}")
        logger.info(f"Training samples: {len(train_indices)}")
        logger.info(f"Validation samples: {len(val_indices)}")
        logger.info(f"Test samples: {len(test_indices)}")
        
        return self.processed_path
    
    def create_torch_dataset(self, output_dir=None):
        """
        Convert HDF5 dataset to PyTorch tensors for easy loading
        
        Args:
            output_dir: Directory to save tensors (if None, use same directory as HDF5)
            
        Returns:
            Path to the output directory
        """
        if output_dir is None:
            output_dir = self.processed_path.parent / 'torch'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Converting dataset to PyTorch tensors in {output_dir}")
        
        # Load the HDF5 file
        with h5py.File(self.processed_path, 'r') as f:
            # Create and save training tensors
            train_inputs = torch.tensor(f['train/inputs'][:], dtype=torch.float32)
            train_outputs = torch.tensor(f['train/outputs'][:], dtype=torch.float32)
            
            # Create and save validation tensors
            val_inputs = torch.tensor(f['val/inputs'][:], dtype=torch.float32)
            val_outputs = torch.tensor(f['val/outputs'][:], dtype=torch.float32)
            
            # Create and save test tensors
            test_inputs = torch.tensor(f['test/inputs'][:], dtype=torch.float32)
            test_outputs = torch.tensor(f['test/outputs'][:], dtype=torch.float32)
        
        # Save tensors
        torch.save(train_inputs, output_dir / 'train_inputs.pt')
        torch.save(train_outputs, output_dir / 'train_outputs.pt')
        torch.save(val_inputs, output_dir / 'val_inputs.pt')
        torch.save(val_outputs, output_dir / 'val_outputs.pt')
        torch.save(test_inputs, output_dir / 'test_inputs.pt')
        torch.save(test_outputs, output_dir / 'test_outputs.pt')
        
        logger.info("PyTorch tensors saved successfully")
        
        return output_dir


def main():
    """Main function to run data preprocessing"""
    parser = argparse.ArgumentParser(description='Preprocess LPBF FEA data for PINN training')
    parser.add_argument('--config', type=str, default='../data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--torch', action='store_true',
                        help='Also create PyTorch tensor datasets')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = LPBFDataPreprocessor(args.config)
    
    # Process all files
    h5_path = preprocessor.process_all_files()
    
    # Optionally create PyTorch datasets
    if args.torch:
        torch_dir = preprocessor.create_torch_dataset()
        print(f"PyTorch datasets saved to {torch_dir}")
    
    print(f"Preprocessing complete. Dataset saved to {h5_path}")


if __name__ == '__main__':
    main()