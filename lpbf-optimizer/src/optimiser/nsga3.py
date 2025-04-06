import numpy as np
import torch
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.factory import get_sampling, get_crossover, get_mutation
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
import h5py
import os


class SurrogateProblem(Problem):
    """
    Multi-objective optimization problem using PINN surrogate
    
    This class wraps the trained PINN model to create a problem
    suitable for multi-objective optimization with pymoo.
    """
    
    def __init__(self, model, param_bounds, objectives, device='cpu'):
        """
        Initialize the surrogate problem
        
        Args:
            model: Trained PINN model
            param_bounds: Dictionary with min/max bounds for each parameter
                Format: {'param_name': [min, max], ...}
            objectives: List of objective names to optimize (must match model outputs)
                Example: ['residual_stress', 'porosity', 'geometric_accuracy']
            device: Device to run model inference on ('cpu' or 'cuda')
        """
        # Extract parameter bounds
        self.n_var = len(param_bounds)
        self.param_names = list(param_bounds.keys())
        self.xl = np.array([param_bounds[p][0] for p in self.param_names])
        self.xu = np.array([param_bounds[p][1] for p in self.param_names])
        
        # Set objectives
        self.n_obj = len(objectives)
        self.objectives = objectives
        
        # The optimization is unconstrained
        self.n_constr = 0
        
        # Store the surrogate model
        self.model = model
        self.device = device
        
        # Call parent constructor
        super().__init__(
            n_var=self.n_var,
            n_obj=self.n_obj,
            n_constr=self.n_constr,
            xl=self.xl,
            xu=self.xu
        )
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the surrogate model for optimization
        
        Args:
            x: Process parameter vectors to evaluate [n_points, n_var]
            out: Output dictionary to store results
            
        Returns:
            Dictionary with objective values
        """
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Normalize inputs if needed
        # This should match the normalization used during training
        
        # Add dummy spatial coordinates and time
        # For optimization, we're usually interested in final state at specific locations
        batch_size = x_tensor.shape[0]
        coords = torch.zeros(batch_size, 3, device=self.device)  # Origin point
        time = torch.ones(batch_size, 1, device=self.device)     # Final time step
        
        # Forward pass through the model
        with torch.no_grad():
            model_input = torch.cat([x_tensor, coords, time], dim=1)
            predictions = self.model(model_input)
        
        # Extract objective values (note: for minimization, so we might need to negate some)
        # We'll keep residual stress and porosity as is (minimize)
        # But for geometric accuracy ratio (GAR), higher is better, so we negate
        objectives = predictions.cpu().numpy()
        
        # Set the objective values for pymoo
        out["F"] = objectives
        
        # For visualization, we might also want to return the predicted values directly
        out["prediction"] = objectives


class NSGAOptimizer:
    """
    Multi-objective optimizer using NSGA-III algorithm
    """
    
    def __init__(self, config_path, model_path):
        """
        Initialize the optimizer
        
        Args:
            config_path: Path to the configuration file
            model_path: Path to the trained PINN model checkpoint
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained PINN model
        self.model = self._load_model(model_path)
        
        # Create the optimization problem
        self.problem = self._create_problem()
        
        # Set up the optimizer
        self.algorithm = self._setup_algorithm()
        
        # Output directory
        self.output_dir = Path(self.config['optimizer']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_model(self, model_path):
        """
        Load the trained PINN model
        
        Args:
            model_path: Path to model checkpoint
            
        Returns:
            Loaded PINN model
        """
        # Import the model class here to avoid circular imports
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pinn'))
        from model import PINN
        
        # Create model with same architecture as during training
        model_config = self.config['model']
        model = PINN(
            in_dim=model_config['input_dim'],
            out_dim=model_config['output_dim'],
            width=model_config['hidden_width'],
            depth=model_config['hidden_depth']
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode
        model.eval()
        model = model.to(self.device)
        
        return model
    
    def _create_problem(self):
        """
        Create the surrogate optimization problem
        
        Returns:
            SurrogateProblem instance
        """
        # Get parameter bounds
        param_bounds = self.config['optimizer']['param_bounds']
        
        # Get objectives
        objectives = self.config['optimizer']['objectives']
        
        # Create the problem
        return SurrogateProblem(
            model=self.model,
            param_bounds=param_bounds,
            objectives=objectives,
            device=self.device
        )
    
    def _setup_algorithm(self):
        """
        Set up the NSGA-III algorithm
        
        Returns:
            Configured NSGA-III algorithm
        """
        # Get optimizer configuration
        opt_config = self.config['optimizer']
        
        # Reference directions using Das-Dennis approach with uniform distribution
        n_partitions = opt_config.get('n_partitions', 12)
        ref_dirs = get_reference_directions(
            "das-dennis", 
            self.problem.n_obj, 
            n_partitions=n_partitions
        )
        
        # Create the algorithm
        algorithm = NSGA3(
            pop_size=opt_config.get('pop_size', 100),
            ref_dirs=ref_dirs,
            sampling=get_sampling("real_random"),
            crossover=get_crossover("real_sbx", prob=0.9, eta=15),
            mutation=get_mutation("real_pm", eta=20),
            eliminate_duplicates=True
        )
        
        return algorithm
    
    def optimize(self):
        """
        Run the optimization process
        
        Returns:
            Optimization results
        """
        # Get termination criteria
        n_gen = self.config['optimizer'].get('n_gen', 100)
        
        # Run the optimization
        print("Starting NSGA-III optimization...")
        res = minimize(
            problem=self.problem,
            algorithm=self.algorithm,
            termination=('n_gen', n_gen),
            seed=1,
            save_history=True,
            verbose=True
        )
        
        # Save the results
        self._save_results(res)
        
        # Visualize results
        self._visualize_results(res)
        
        return res
    
    def _save_results(self, res):
        """
        Save optimization results
        
        Args:
            res: Optimization results
        """
        # Extract optimal solutions (process parameters)
        X = res.X  # Pareto optimal process parameters
        F = res.F  # Corresponding objective values
        
        # Save to HDF5
        with h5py.File(self.output_dir / 'pareto_solutions.h5', 'w') as f:
            # Save process parameters
            f.create_dataset('parameters', data=X)
            # Save corresponding objective values
            f.create_dataset('objectives', data=F)
            # Save parameter names
            dt = h5py.special_dtype(vlen=str)
            param_names = np.array(self.problem.param_names, dtype=object)
            obj_names = np.array(self.problem.objectives, dtype=object)
            f.create_dataset('parameter_names', data=param_names, dtype=dt)
            f.create_dataset('objective_names', data=obj_names, dtype=dt)
            
            # Store metadata
            f.attrs['n_solutions'] = len(X)
            f.attrs['n_parameters'] = X.shape[1]
            f.attrs['n_objectives'] = F.shape[1]
        
        # Save as CSV for easy viewing
        import pandas as pd
        param_df = pd.DataFrame(X, columns=self.problem.param_names)
        obj_df = pd.DataFrame(F, columns=self.problem.objectives)
        result_df = pd.concat([param_df, obj_df], axis=1)
        result_df.to_csv(self.output_dir / 'pareto_solutions.csv', index=False)
    
    def _visualize_results(self, res):
        """
        Visualize optimization results
        
        Args:
            res: Optimization results
        """
        # Extract objective values
        F = res.F
        obj_names = self.problem.objectives
        
        # Plot Pareto front
        if F.shape[1] == 2:
            # 2D Pareto front
            plt.figure(figsize=(10, 8))
            plt.scatter(F[:, 0], F[:, 1], s=30)
            plt.xlabel(obj_names[0])
            plt.ylabel(obj_names[1])
            plt.title('Pareto Front')
            plt.grid(True)
            plt.savefig(self.output_dir / 'pareto_front_2d.png')
            plt.close()
        
        elif F.shape[1] == 3:
            # 3D Pareto front
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(F[:, 0], F[:, 1], F[:, 2], s=30)
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])
            ax.set_zlabel(obj_names[2])
            ax.set_title('Pareto Front')
            plt.savefig(self.output_dir / 'pareto_front_3d.png')
            plt.close()
        
        # Plot parallel coordinates for more than 3 objectives
        from pymoo.visualization.pcp import PCP
        pcp = PCP(tight_layout=True, legend=(True, {'loc': 'upper left'}))
        pcp.add(F, color="navy")
        pcp.show()
        plt.savefig(self.output_dir / 'pareto_pcp.png')
        plt.close()
        
        # Plot the convergence
        n_gen = len(res.history)
        plt.figure(figsize=(10, 6))
        
        # Extract convergence metrics
        convergence = [res.history[i].opt.get("F").min(axis=0) for i in range(n_gen)]
        convergence = np.array(convergence)
        
        # Plot for each objective
        for i in range(convergence.shape[1]):
            plt.plot(np.arange(n_gen), convergence[:, i], label=obj_names[i])
        
        plt.xlabel('Generation')
        plt.ylabel('Minimum Objective Value')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'convergence.png')
        plt.close()


def main():
    """Main function to run optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run NSGA-III optimization with PINN surrogate')
    parser.add_argument('--config', type=str, default='../../data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained PINN model checkpoint')
    
    args = parser.parse_args()
    
    # Create and run optimizer
    optimizer = NSGAOptimizer(args.config, args.model)
    results = optimizer.optimize()
    
    # Print summary
    print("\nOptimization complete!")
    print(f"Number of solutions in Pareto front: {len(results.X)}")
    print(f"Results saved to: {optimizer.output_dir}")


if __name__ == '__main__':
    main()