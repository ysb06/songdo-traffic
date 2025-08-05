"""
Performance testing script for RNN model on traffic prediction
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from metr.dataloader import TrafficDataModule, collate_simple
from metr_val.model import BasicRNN

logger = logging.getLogger(__name__)


def load_traffic_data(data_dir: str = "../datasets/metr-imc") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load training and test traffic data from the configured directory
    
    Args:
        data_dir: Directory containing the traffic data files
        
    Returns:
        Tuple of (training_df, test_df)
    """
    data_path = Path(data_dir)
    
    # Look for CSV files (adjust based on actual file structure)
    train_files = list(data_path.glob("*train*.csv")) + list(data_path.glob("*training*.csv"))
    test_files = list(data_path.glob("*test*.csv"))
    
    if not train_files:
        # Try alternative patterns
        train_files = list(data_path.glob("train_*.csv")) + list(data_path.glob("training_*.csv"))
        
    if not test_files:
        test_files = list(data_path.glob("test_*.csv"))
    
    if not train_files or not test_files:
        # Try loading from subdirectories
        for subdir in data_path.iterdir():
            if subdir.is_dir():
                train_files.extend(list(subdir.glob("*train*.csv")))
                train_files.extend(list(subdir.glob("*training*.csv")))
                test_files.extend(list(subdir.glob("*test*.csv")))
    
    if not train_files:
        raise FileNotFoundError(f"No training data files found in {data_path}")
    if not test_files:
        raise FileNotFoundError(f"No test data files found in {data_path}")
    
    # Load the first available files
    train_file = train_files[0]
    test_file = test_files[0]
    
    logger.info(f"Loading training data from: {train_file}")
    logger.info(f"Loading test data from: {test_file}")
    
    # Load data with datetime index
    train_df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    test_df = pd.read_csv(test_file, index_col=0, parse_dates=True)
    
    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")
    logger.info(f"Training data date range: {train_df.index.min()} to {train_df.index.max()}")
    logger.info(f"Test data date range: {test_df.index.min()} to {test_df.index.max()}")
    
    return train_df, test_df


def create_data_module(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_length: int = 24,
    batch_size: int = 32,
    valid_split_datetime: str = "2024-08-01 00:00:00",
    target_sensors: Optional[list] = None
) -> TrafficDataModule:
    """
    Create TrafficDataModule for training
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe  
        seq_length: Sequence length for time series
        batch_size: Batch size for training
        valid_split_datetime: Datetime to split train/validation
        target_sensors: List of target sensors (None for all)
        
    Returns:
        Configured TrafficDataModule
    """
    data_module = TrafficDataModule(
        training_df=train_df,
        test_df=test_df,
        seq_length=seq_length,
        batch_size=batch_size,
        num_workers=0,  # Set to 0 for compatibility
        shuffle_training=True,
        collate_fn=collate_simple,
        valid_split_datetime=valid_split_datetime,
        training_target_sensor=target_sensors,
        scale_strictly=False  # For faster processing
    )
    
    return data_module


def test_rnn_performance(
    data_dir: str = "../datasets/metr-imc",
    seq_length: int = 24,
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 15,
    target_sensors: Optional[list] = None,
    results_dir: str = "./rnn_results",
    **kwargs
) -> Dict[str, Any]:
    """
    Test RNN model performance on traffic prediction task
    
    Args:
        data_dir: Directory containing traffic data
        seq_length: Input sequence length
        hidden_size: RNN hidden size
        num_layers: Number of RNN layers
        learning_rate: Learning rate
        batch_size: Batch size
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        target_sensors: Target sensors to use (None for all)
        results_dir: Directory to save results
        **kwargs: Additional arguments
        
    Returns:
        Dictionary with test results
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting RNN performance test...")
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    try:
        # Load data
        logger.info("Loading traffic data...")
        train_df, test_df = load_traffic_data(data_dir)
        
        # Create data module
        logger.info("Creating data module...")
        data_module = create_data_module(
            train_df=train_df,
            test_df=test_df,
            seq_length=seq_length,
            batch_size=batch_size,
            target_sensors=target_sensors
        )
        
        # Create model
        logger.info("Creating RNN model...")
        model = BasicRNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=1,
            learning_rate=learning_rate
        )
        
        # Setup callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            verbose=True,
            mode='min'
        )
        
        checkpoint = ModelCheckpoint(
            dirpath=results_dir,
            filename='best_rnn_model-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=1
        )
        
        # Setup logger
        csv_logger = CSVLogger(results_dir, name='rnn_experiment')
        
        # Create trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stopping, checkpoint],
            logger=csv_logger,
            enable_progress_bar=True,
            log_every_n_steps=10,
            accelerator='auto'  # Use GPU if available
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        
        # Test model
        logger.info("Starting testing...")
        test_results = trainer.test(model, data_module)
        
        # Save results
        results = {
            'model_params': {
                'seq_length': seq_length,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'learning_rate': learning_rate,
                'batch_size': batch_size
            },
            'test_metrics': test_results[0] if test_results else {},
            'best_model_path': checkpoint.best_model_path,
            'training_completed': True
        }
        
        logger.info("RNN performance test completed successfully!")
        logger.info(f"Results saved to: {results_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error during RNN performance test: {str(e)}")
        return {
            'error': str(e),
            'training_completed': False
        }


def main():
    """Main function to run RNN performance test"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test RNN model performance on traffic data')
    parser.add_argument('--data_dir', type=str, default='../datasets/metr-imc',
                       help='Directory containing traffic data')
    parser.add_argument('--seq_length', type=int, default=24,
                       help='Input sequence length')
    parser.add_argument('--hidden_size', type=int, default=64,
                       help='RNN hidden size')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of RNN layers')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum training epochs')
    parser.add_argument('--patience', type=int, default=15,
                       help='Early stopping patience')
    parser.add_argument('--results_dir', type=str, default='./rnn_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run test
    results = test_rnn_performance(**vars(args))
    
    if results.get('training_completed', False):
        print("\n" + "="*50)
        print("RNN PERFORMANCE TEST COMPLETED")
        print("="*50)
        print(f"Best model saved to: {results.get('best_model_path', 'N/A')}")
        print(f"Results directory: {args.results_dir}")
        
        test_metrics = results.get('test_metrics', {})
        if test_metrics:
            print("\nTest Metrics:")
            for key, value in test_metrics.items():
                print(f"  {key}: {value:.4f}")
    else:
        print(f"\nTest failed with error: {results.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()