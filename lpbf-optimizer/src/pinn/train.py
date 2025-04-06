import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import yaml
import h5py
from pathlib import Path

from model import PINN
from physics import compute_physics_loss


class PINNTrainer:
    """
    Trainer for the Physics-Informed Neural Network for LPBF process optimization
    """
    def __init__(self, config_path):
        """
        Initialize the trainer with configuration
        
        Args:
            config_path (str): Path to the configuration YAML file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load material properties
        self.mat_props = self.config['material_properties']
        
        # Set up directories
        self.setup_directories()
        
        # Initialize model
        self.model = self._build_model()
        
        # Set up optimizer
        self.optimizer = self._build_optimizer()
        
        # Set up learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Track training metrics
        self.metrics = {
            'train_loss': [],
            'val_loss': [],
            'data_loss': [],
            'physics_loss': []
        }
        
        # Current epoch
        self.current_epoch = 0
        
    def setup_directories(self):
        """Create necessary directories for saving results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(self.config['training']['output_dir']) / timestamp
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.log_dir = self.run_dir / 'logs'
        self.plot_dir = self.run_dir / 'plots'
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        self.plot_dir.mkdir(exist_ok=True)
        
        # Save config to run directory
        with open(self.run_dir / 'config.yaml', 'w') as f:
            yaml.dump(self.config, f)
    
    def _build_model(self):
        """Initialize the PINN model"""
        model_config = self.config['model']
        model = PINN(
            in_dim=model_config['input_dim'],
            out_dim=model_config['output_dim'],
            width=model_config['hidden_width'],
            depth=model_config['hidden_depth']
        )
        model = model.to(self.device)
        return model
    
    def _build_optimizer(self):
        """Set up the optimizer"""
        optim_config = self.config['training']['optimizer']
        if optim_config['type'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=optim_config['learning_rate'],
                weight_decay=optim_config.get('weight_decay', 0)
            )
        else:
            raise NotImplementedError(f"Optimizer {optim_config['type']} not implemented.")
        return optimizer
    
    def _build_scheduler(self):
        """Set up the learning rate scheduler"""
        sched_config = self.config['training'].get('scheduler', {})
        if not sched_config or sched_config.get('type', '').lower() == 'none':
            return None
        
        if sched_config['type'].lower() == 'reducelronplateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.1),
                patience=sched_config.get('patience', 10),
                verbose=True
            )
        elif sched_config['type'].lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', 100),
                eta_min=sched_config.get('eta_min', 0)
            )
        else:
            raise NotImplementedError(f"Scheduler {sched_config['type']} not implemented.")
        return scheduler
    
    def load_data(self):
        """
        Load training and validation data from processed HDF5 files
        """
        data_config = self.config['data']
        data_file = Path(data_config['processed_data_path'])
        
        # Load data from HDF5 file
        with h5py.File(data_file, 'r') as f:
            # Process parameters (scan vectors)
            S_train = torch.tensor(f['train/scan_vectors'][:], dtype=torch.float32)
            S_val = torch.tensor(f['val/scan_vectors'][:], dtype=torch.float32)
            
            # Coordinate and time information
            coords_train = torch.tensor(f['train/coordinates'][:], dtype=torch.float32)
            coords_val = torch.tensor(f['val/coordinates'][:], dtype=torch.float32)
            time_train = torch.tensor(f['train/time'][:], dtype=torch.float32)
            time_val = torch.tensor(f['val/time'][:], dtype=torch.float32)
            
            # Ground truth values (from FEA)
            y_train = torch.tensor(f['train/outputs'][:], dtype=torch.float32)
            y_val = torch.tensor(f['val/outputs'][:], dtype=torch.float32)
        
        # Move data to device
        self.train_data = {
            'S': S_train.to(self.device),
            'coords': coords_train.to(self.device),
            'time': time_train.to(self.device),
            'y': y_train.to(self.device)
        }
        
        self.val_data = {
            'S': S_val.to(self.device),
            'coords': coords_val.to(self.device),
            'time': time_val.to(self.device),
            'y': y_val.to(self.device)
        }
        
        print(f"Loaded {len(S_train)} training samples and {len(S_val)} validation samples")
    
    def train_step(self, S_batch, coords_batch, t_batch, y_batch):
        """
        Perform one training step
        
        Args:
            S_batch: Process parameters batch
            coords_batch: Spatial coordinates batch
            t_batch: Time coordinates batch
            y_batch: Ground truth outputs batch
        
        Returns:
            dict: Loss components
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        model_input = torch.cat([S_batch, coords_batch, t_batch], dim=1)
        y_pred = self.model(model_input)
        
        # Data loss
        mse_loss = nn.MSELoss()
        data_loss = mse_loss(y_pred, y_batch)
        
        # Physics loss
        physics_loss = compute_physics_loss(
            self.model, 
            S_batch, 
            coords_batch, 
            t_batch, 
            self.mat_props,
            lambda_heat=self.config['training']['lambda_heat'],
            lambda_stress=self.config['training']['lambda_stress']
        )
        
        # Total loss
        total_loss = data_loss + physics_loss
        
        # Backward pass
        total_loss.backward()
        
        # Clip gradients
        if self.config['training'].get('clip_grad', False):
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config['training'].get('clip_value', 1.0)
            )
        
        # Optimizer step
        self.optimizer.step()
        
        return {
            'total': total_loss.item(),
            'data': data_loss.item(),
            'physics': physics_loss.item()
        }
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Args:
            epoch: Current epoch number
        
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()
        
        # Create data loader
        batch_size = self.config['training']['batch_size']
        dataset_size = len(self.train_data['S'])
        
        # Calculate number of batches
        n_batches = dataset_size // batch_size + (1 if dataset_size % batch_size != 0 else 0)
        
        # Initialize losses
        epoch_loss = 0
        epoch_data_loss = 0
        epoch_physics_loss = 0
        
        # Shuffle indices
        indices = torch.randperm(dataset_size)
        
        # Train over batches
        for i in range(n_batches):
            # Get batch indices
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, dataset_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            S_batch = self.train_data['S'][batch_indices]
            coords_batch = self.train_data['coords'][batch_indices]
            t_batch = self.train_data['time'][batch_indices]
            y_batch = self.train_data['y'][batch_indices]
            
            # Training step
            losses = self.train_step(S_batch, coords_batch, t_batch, y_batch)
            
            # Update epoch losses
            epoch_loss += losses['total']
            epoch_data_loss += losses['data']
            epoch_physics_loss += losses['physics']
            
            # Print progress
            if (i + 1) % self.config['training'].get('print_freq', 10) == 0:
                print(f"Epoch [{epoch+1}/{self.config['training']['n_epochs']}] "
                      f"Batch [{i+1}/{n_batches}] "
                      f"Loss: {losses['total']:.6f} "
                      f"(Data: {losses['data']:.6f}, Physics: {losses['physics']:.6f})")
        
        # Calculate average losses
        avg_loss = epoch_loss / n_batches
        avg_data_loss = epoch_data_loss / n_batches
        avg_physics_loss = epoch_physics_loss / n_batches
        
        # Update metrics
        self.metrics['train_loss'].append(avg_loss)
        self.metrics['data_loss'].append(avg_data_loss)
        self.metrics['physics_loss'].append(avg_physics_loss)
        
        return avg_loss
    
    def validate(self):
        """
        Validate the model on validation data
        
        Returns:
            float: Validation loss
        """
        self.model.eval()
        
        # Create data loader
        batch_size = self.config['training']['batch_size'] * 2  # Larger batch for validation
        dataset_size = len(self.val_data['S'])
        
        # Calculate number of batches
        n_batches = dataset_size // batch_size + (1 if dataset_size % batch_size != 0 else 0)
        
        # Initialize loss
        val_loss = 0
        
        with torch.no_grad():
            # Validate over batches
            for i in range(n_batches):
                # Get batch indices
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, dataset_size)
                
                # Get batch data
                S_batch = self.val_data['S'][start_idx:end_idx]
                coords_batch = self.val_data['coords'][start_idx:end_idx]
                t_batch = self.val_data['time'][start_idx:end_idx]
                y_batch = self.val_data['y'][start_idx:end_idx]
                
                # Forward pass
                model_input = torch.cat([S_batch, coords_batch, t_batch], dim=1)
                y_pred = self.model(model_input)
                
                # Compute loss
                mse_loss = nn.MSELoss()
                batch_loss = mse_loss(y_pred, y_batch)
                
                # Update validation loss
                val_loss += batch_loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / n_batches
        
        # Update metrics
        self.metrics['val_loss'].append(avg_val_loss)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            loss: Current loss
            is_best: Whether this is the best model so far
        """
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{epoch+1}.pt"
        best_model_path = self.checkpoint_dir / "best_model.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'metrics': self.metrics
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if applicable
        if is_best:
            torch.save(checkpoint, best_model_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.metrics = checkpoint['metrics']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}")
    
    def plot_metrics(self):
        """Plot training and validation metrics"""
        # Plot loss curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['train_loss'], label='Train Loss')
        plt.plot(self.metrics['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_dir / 'loss_curves.png')
        
        # Plot data vs physics loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['data_loss'], label='Data Loss')
        plt.plot(self.metrics['physics_loss'], label='Physics Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Data and Physics Loss Components')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plot_dir / 'loss_components.png')
    
    def train(self):
        """Main training loop"""
        print("Beginning training...")
        
        # Load data
        self.load_data()
        
        # Training parameters
        n_epochs = self.config['training']['n_epochs']
        best_val_loss = float('inf')
        
        # Train for n_epochs
        for epoch in range(self.current_epoch, n_epochs):
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Print epoch results
            print(f"Epoch [{epoch+1}/{n_epochs}] "
                  f"Train Loss: {train_loss:.6f}, "
                  f"Validation Loss: {val_loss:.6f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Save checkpoint periodically and for best model
            if (epoch + 1) % self.config['training']['checkpoint_freq'] == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
            
            # Plot metrics
            if (epoch + 1) % self.config['training']['plot_freq'] == 0:
                self.plot_metrics()
            
            # Update current epoch
            self.current_epoch = epoch + 1
        
        # Final plots and checkpoints
        self.plot_metrics()
        self.save_checkpoint(n_epochs - 1, val_loss)
        
        print("Training complete!")


def main():
    """Main function to start training"""
    parser = argparse.ArgumentParser(description='Train PINN for LPBF optimization')
    parser.add_argument('--config', type=str, default='../../data/params.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file for resuming training')
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = PINNTrainer(args.config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()