import numpy as np
import torch
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import h5py
import time

# For Bayesian optimization with Ax/BoTorch
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.tutorials.cbo_utils import normalize
import ax.modelbridge.registry as registry


class BayesianOptimizer:
    """
    Bayesian optimization for LPBF process parameter tuning using the PINN surrogate model.
    
    This optimizer uses Ax/BoTorch which implements Bayesian Optimization with
    Thompson sampling. It's useful for more efficiently exploring the parameter space
    compared to genetic algorithms when the number of evaluations is limited.
    """
    
    def __init__(self, config_path, model_path):
        """
        Initialize the Bayesian optimizer
        
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
    
    def _create_evaluation_function(self, objective_idx=0):
        """
        Create an evaluation function for the objective
        
        Args:
            objective_idx: Index of the objective to optimize (default: 0 for residual stress)
        
        Returns:
            Evaluation function for Ax
        """
        # Define the evaluation function
        def evaluate(parameters):
            # Convert parameters to tensor
            param_values = [parameters[p] for p in self.param_names]
            x = torch.tensor([param_values], dtype=torch.float32, device=self.device)
            
            # Add dummy spatial coordinates and time
            batch_size = x.shape[0]
            coords = torch.zeros(batch_size, 3, device=self.device)  # Origin point
            time = torch.ones(batch_size, 1, device=self.device)     # Final time step
            
            # Forward pass through the model
            with torch.no_grad():
                model_input = torch.cat([x, coords, time], dim=1)
                predictions = self.model(model_input)
            
            # Extract target objective value
            obj_value = predictions[0, objective_idx].item()
            
            # For stochastic models, we could return SEM, but our surrogate is deterministic
            return {"objective": (obj_value, 0.0)}
        
        return evaluate
    
    def _create_params_and_objectives(self):
        """
        Create parameters and objectives definitions for Ax
        
        Returns:
            Tuple of parameters and objectives lists for Ax
        """
        # Get parameter bounds
        param_bounds = self.config['optimizer']['param_bounds']
        self.param_names = list(param_bounds.keys())
        
        # Define parameters
        parameters = []
        for name in self.param_names:
            bounds = param_bounds[name]
            parameters.append({
                "name": name,
                "type": "range",
                "bounds": bounds,
                "value_type": "float"
            })
        
        # Define objective - for Bayesian optimization we typically
        # focus on one objective at a time
        objective_name = self.config['optimizer']['objectives'][0]
        objectives = [
            {
                "name": "objective",
                "type": "minimize",  # Assume we want to minimize (e.g., residual stress)
                "properties": {
                    "objective_name": objective_name
                }
            }
        ]
        
        return parameters, objectives
    
    def run_single_objective_optimization(self, objective_idx=0, n_trials=50):
        """
        Run Bayesian optimization for a single objective
        
        Args:
            objective_idx: Index of the objective to optimize
            n_trials: Number of trials to run
            
        Returns:
            Optimization results
        """
        # Create parameters and objectives definitions
        parameters, objectives = self._create_params_and_objectives()
        
        # Get the objective name
        objective_name = self.config['optimizer']['objectives'][objective_idx]
        print(f"Optimizing for {objective_name}")
        
        # Create evaluation function
        evaluation_function = self._create_evaluation_function(objective_idx)
        
        # Run optimization
        print(f"Running Bayesian optimization with {n_trials} trials...")
        start_time = time.time()
        
        best_parameters, values, experiment, model = optimize(
            parameters=parameters,
            evaluation_function=evaluation_function,
            objective_name="objective",
            minimize=True,  # Minimize the objective
            total_trials=n_trials,
            random_seed=self.config['optimizer'].get('seed', 42)
        )
        
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds")
        
        # Print results
        print("\nBest parameters:")
        for param, value in best_parameters.items():
            print(f"  {param}: {value}")
        print(f"Best objective value: {values[0]['objective']}")
        
        # Save results
        self._save_results(experiment, model, objective_name)
        
        return best_parameters, values, experiment, model
    
    def run_multi_objective_optimization(self, n_trials=50):
        """
        Run a series of single-objective optimizations for each objective
        
        This is a simple approach to multi-objective optimization with Bayesian methods.
        For more sophisticated approaches, MOO extensions to BoTorch could be used.
        
        Args:
            n_trials: Number of trials per objective
            
        Returns:
            List of results for each objective
        """
        objectives = self.config['optimizer']['objectives']
        results = []
        
        for i, obj_name in enumerate(objectives):
            print(f"\nOptimizing for objective {i+1}/{len(objectives)}: {obj_name}")
            result = self.run_single_objective_optimization(i, n_trials)
            results.append(result)
        
        return results
    
    def _save_results(self, experiment, model, objective_name):
        """
        Save optimization results
        
        Args:
            experiment: Ax experiment
            model: Ax model
            objective_name: Name of the objective
        """
        # Get the data from the experiment
        data = experiment.fetch_data()
        df = data.df
        
        # Save raw data to CSV
        csv_path = self.output_dir / f"bayesopt_{objective_name}.csv"
        df.to_csv(csv_path, index=False)
        
        # Extract parameters and objective values
        parameters = []
        objective_values = []
        
        for _, row in df.iterrows():
            params = {}
            for param in self.param_names:
                params[param] = row[f"arm_parameters.{param}"]
            parameters.append(list(params.values()))
            objective_values.append(row["mean"])
        
        # Convert to numpy arrays
        X = np.array(parameters)
        Y = np.array(objective_values).reshape(-1, 1)
        
        # Save to HDF5
        with h5py.File(self.output_dir / f"bayesopt_{objective_name}.h5", 'w') as f:
            # Save process parameters
            f.create_dataset('parameters', data=X)
            # Save corresponding objective values
            f.create_dataset('objectives', data=Y)
            # Save parameter names
            dt = h5py.special_dtype(vlen=str)
            param_names = np.array(self.param_names, dtype=object)
            f.create_dataset('parameter_names', data=param_names, dtype=dt)
            # Save objective name
            obj_name = np.array([objective_name], dtype=object)
            f.create_dataset('objective_names', data=obj_name, dtype=dt)
            
            # Store metadata
            f.attrs['n_trials'] = len(X)
            f.attrs['n_parameters'] = X.shape[1]
            f.attrs['best_objective'] = Y.min()
            
            # Store best parameters
            best_idx = Y.argmin()
            f.attrs['best_params'] = X[best_idx]
        
        # Generate plots
        self._plot_convergence(df, objective_name)
        self._plot_importance(model, objective_name)
    
    def _plot_convergence(self, df, objective_name):
        """
        Plot optimization convergence
        
        Args:
            df: DataFrame with optimization data
            objective_name: Name of the objective
        """
        plt.figure(figsize=(10, 6))
        best_so_far = np.minimum.accumulate(df['mean'])
        plt.plot(range(1, len(best_so_far) + 1), best_so_far)
        plt.xlabel('Trial')
        plt.ylabel(f'Best {objective_name} so far')
        plt.title('Optimization Convergence')
        plt.grid(True)
        plt.savefig(self.output_dir / f"bayesopt_convergence_{objective_name}.png")
        plt.close()
    
    def _plot_importance(self, model, objective_name):
        """
        Plot parameter importance if a model is available
        
        Args:
            model: Ax model
            objective_name: Name of the objective
        """
        if model is not None and hasattr(model, 'feature_importances'):
            plt.figure(figsize=(10, 6))
            importances = model.feature_importances()
            if importances is not None:
                params = list(importances.keys())
                values = list(importances.values())
                
                # Sort by importance
                sorted_indices = np.argsort(values)
                sorted_params = [params[i] for i in sorted_indices]
                sorted_values = [values[i] for i in sorted_indices]
                
                plt.barh(sorted_params, sorted_values)
                plt.xlabel('Relative Importance')
                plt.title(f'Parameter Importance for {objective_name}')
                plt.tight_layout()
                plt.savefig(self.output_dir / f"bayesopt_importance_{objective_name}.png")
                plt.close()


def main():
    """Main function to run Bayesian optimization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Bayesian optimization with PINN surrogate')
    parser.add_argument('--config', type=str, default='../../data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained PINN model checkpoint')
    parser.add_argument('--objective', type=int, default=0,
                        help='Index of the objective to optimize (default: 0 for residual stress)')
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--multi', action='store_true',
                        help='Run multi-objective optimization (sequential)')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = BayesianOptimizer(args.config, args.model)
    
    # Run optimization
    if args.multi:
        results = optimizer.run_multi_objective_optimization(args.trials)
    else:
        results = optimizer.run_single_objective_optimization(args.objective, args.trials)
    
    # Print summary
    print("\nOptimization complete!")
    print(f"Results saved to: {optimizer.output_dir}")


if __name__ == '__main__':
    main()