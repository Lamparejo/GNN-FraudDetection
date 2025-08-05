"""
Main fraud detection system using Graph Neural Networks.

This is the main entry point of the system, coordinating
the complete pipeline from data loading to model training
and evaluation.
"""

import argparse
import yaml
from pathlib import Path
import torch
from loguru import logger

from src.utils import Config, Timer, device_manager
from src.data import DataPipeline
from src.models import ModelFactory
from src.training import GNNTrainer


class FraudDetectionSystem:
    """
    Main fraud detection system.
    
    Coordinates the complete machine learning pipeline for fraud
    detection using Graph Neural Networks.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.data_pipeline = None
        self.model = None
        self.trainer = None
        
        logger.info("Fraud detection system initialized")
        logger.info(f"Device: {device_manager.device}")
    
    def setup_data_pipeline(self) -> None:
        """Set up data pipeline."""
        with Timer("Data pipeline setup"):
            # Use only the base data directory
            data_path = str(Path("ieee-fraud-detection"))
            self.data_pipeline = DataPipeline(data_path)
            logger.info("Data pipeline configured")
    
    def load_and_process_data(self) -> None:
        """Load and process data."""
        if self.data_pipeline is None:
            self.setup_data_pipeline()
        
        assert self.data_pipeline is not None, "Data pipeline must be configured"
        
        sample_size = self.config.get('data.sample_size')
        self.train_graph, self.test_graph = self.data_pipeline.process_data(sample_size)
        
        # Log data information
        data_summary = self.data_pipeline.get_data_summary()
        logger.info(f"Data processed: {data_summary}")
    
    def create_model(self) -> None:
        """Create the GNN model."""
        with Timer("Model creation"):
            model_config = self.config.model
            
            # Determine input dimension and model type
            if hasattr(self.train_graph, 'x_dict'):
                input_dim = self.train_graph['transaction'].x.shape[1]
                metadata = (self.train_graph.node_types, self.train_graph.edge_types)
                # For heterogeneous graphs, use heterogeneous model
                model_type = 'HETERO'
                logger.info("Detected heterogeneous graph, using HeteroGNN model")
            else:
                input_dim = self.train_graph.x.shape[1]
                metadata = None
                model_type = model_config.get('type', 'GraphSAGE')
            
            # Create model
            if model_type == 'HETERO':
                # Specific parameters for heterogeneous model
                model_params = {
                    'model_type': 'HETERO',
                    'input_dim': input_dim,  # Still needed for factory, but not used
                    'metadata': metadata,
                    'hidden_dim': model_config.get('hidden_dim', 256),
                    'num_classes': 2,
                    'num_layers': model_config.get('num_layers', 3),
                    'dropout': model_config.get('dropout', 0.3),
                    'model_type_inner': 'SAGE'  # Force SAGE to work
                }
            else:
                # Parameters for homogeneous models  
                model_params = {
                    'model_type': model_type,
                    'input_dim': input_dim,
                    'hidden_dim': model_config.get('hidden_dim', 256),
                    'num_layers': model_config.get('num_layers', 3),
                    'dropout': model_config.get('dropout', 0.3),
                    'metadata': metadata
                }
                
                # Add extra parameters avoiding duplication
                for key, value in model_config.items():
                    if key not in model_params:
                        model_params[key] = value
            
            self.model = ModelFactory.create_model(**model_params)
            
            logger.info(f"Model created: {self.model.__class__.__name__}")
            # Don't count uninitialized lazy parameters
            try:
                param_count = sum(p.numel() for p in self.model.parameters() if p.data.numel() > 0)
                logger.info(f"Initialized parameters: {param_count:,}")
            except Exception:
                logger.info("Parameters will be initialized on first forward pass")
    
    def train_model(self) -> dict:
        """Train the model."""
        if self.model is None:
            self.create_model()
        
        assert self.model is not None, "Model must be created"
        
        with Timer("Model training"):
            # Set up trainer
            self.trainer = GNNTrainer(
                model=self.model,
                data=self.train_graph,
                config=self.config.training
            )
            
            # Train model
            epochs = self.config.get('training.epochs', 100)
            history = self.trainer.train(epochs=epochs, verbose=True)
            
            # Log results
            summary = self.trainer.get_training_summary()
            logger.info(f"Training completed: {summary}")
            
            return history
    
    def evaluate_model(self) -> dict:
        """Evaluate the model on test set."""
        if self.trainer is None:
            raise ValueError("Model must be trained before evaluation")
        
        with Timer("Model evaluation"):
            # TODO: Implement evaluation on test set
            # For now, return training summary
            return self.trainer.get_training_summary()
    
    def run_full_pipeline(self) -> dict:
        """
        Execute the complete pipeline.
        
        Returns:
            Training and evaluation results
        """
        logger.info("Starting complete fraud detection pipeline")
        
        results = {}
        
        try:
            # 1. Load and process data
            logger.info("Step 1: Data loading and processing")
            self.load_and_process_data()
            results['data_processing'] = 'success'
            
            # 2. Create model
            logger.info("Step 2: Model creation")
            self.create_model()
            results['model_creation'] = 'success'
            
            # 3. Train model
            logger.info("Step 3: Model training")
            training_history = self.train_model()
            results['training'] = training_history
            
            # 4. Evaluate model
            logger.info("Step 4: Model evaluation")
            evaluation_results = self.evaluate_model()
            results['evaluation'] = evaluation_results
            
            logger.info("Complete pipeline executed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {"error": str(e)}
        
        return results
    
    def predict(self, new_data) -> tuple:
        """
        Make predictions on new data.
        
        Args:
            new_data: New data for prediction
            
        Returns:
            Tuple containing (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model must be loaded or trained before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            if hasattr(new_data, 'x_dict'):  # Heterogeneous graph
                logits = self.model(new_data.x_dict, new_data.edge_index_dict)
            else:  # Homogeneous graph
                logits = self.model(new_data.x, new_data.edge_index)
            
            probabilities = torch.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)
            
        return predictions, probabilities
    
    def save_system(self, save_path: str) -> None:
        """Save the complete system."""
        if self.trainer is None:
            raise ValueError("System must be trained before saving")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.trainer.save_model(str(save_dir / "model.pt"))
        
        # Save configuration
        config_path = save_dir / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(self.config._config, f, default_flow_style=False)
        
        logger.info(f"System saved to {save_dir}")
    
    @classmethod
    def load_system(cls, load_path: str):
        """Load saved system."""
        load_dir = Path(load_path)
        
        # Load configuration
        config_path = load_dir / "config.yaml"
        system = cls(str(config_path))
        
        # TODO: Implement complete model loading
        logger.info(f"System loaded from {load_dir}")
        
        return system


def main():
    """Main function for command line execution."""
    parser = argparse.ArgumentParser(
        description="Fraud Detection System with Graph Neural Networks"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'predict', 'full'],
        default='full',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        help='Custom path to data'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        help='Path to saved model (for predict mode)'
    )
    
    parser.add_argument(
        '--output-path',
        type=str,
        default='results',
        help='Path to save results'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        help='Sample size for development'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = FraudDetectionSystem(args.config)
        
        # Apply command line configurations
        if args.sample_size:
            system.config.set('data.sample_size', args.sample_size)
        
        # Execute selected mode
        if args.mode == 'full':
            results = system.run_full_pipeline()
            system.save_system(args.output_path)
            
        elif args.mode == 'train':
            system.load_and_process_data()
            system.train_model()
            system.save_system(args.output_path)
            
        elif args.mode == 'evaluate':
            if args.model_path:
                system = FraudDetectionSystem.load_system(args.model_path)
            system.evaluate_model()
            
        elif args.mode == 'predict':
            if not args.model_path:
                raise ValueError("Model path is required for predict mode")
            
            system = FraudDetectionSystem.load_system(args.model_path)
            # TODO: Implement new data loading and prediction
            
        logger.info("Execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Execution error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
