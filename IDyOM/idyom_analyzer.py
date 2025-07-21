#!/usr/bin/env python3
"""
Standalone IDyOM Analyzer Script

This script provides a functional interface to IDyOM without requiring Docker or Jupyter.
It uses the existing py2lisp modules to configure and run IDyOM experiments.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Union, Optional
import datetime

# Add the py2lisp module to path if it's not installed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'py2lisp'))

from py2lisp.run import IDyOMExperiment
from py2lisp.export import Export
from py2lisp.extract import ExperimentInfo

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class IDyOMAnalyzer:
    """
    A functional wrapper for IDyOM analysis that doesn't require Docker or Jupyter.
    """
    
    def __init__(self, 
                 test_dataset_path: str,
                 pretrain_dataset_path: Optional[str] = None,
                 output_dir: Optional[str] = None,
                 experiment_name: Optional[str] = None,
                 overwrite: bool = False):
        """
        Initialize the IDyOM analyzer.
        
        Args:
            test_dataset_path: Path to test dataset (MIDI or Kern files)
            pretrain_dataset_path: Optional path to pretraining dataset
            output_dir: Directory to store results (default: ./idyom_results)
            experiment_name: Name for this experiment (default: timestamp)
            overwrite: If True, overwrite existing experiments; if False, create unique names
        """
        self.test_dataset_path = Path(test_dataset_path)
        self.pretrain_dataset_path = Path(pretrain_dataset_path) if pretrain_dataset_path else None
        self.output_dir = Path(output_dir) if output_dir else Path('./idyom_results')
        self.experiment_name = experiment_name
        self.overwrite = overwrite
        
        # Validate input paths
        if not self.test_dataset_path.exists():
            raise FileNotFoundError(f"Test dataset path not found: {self.test_dataset_path}")
        
        if self.pretrain_dataset_path and not self.pretrain_dataset_path.exists():
            raise FileNotFoundError(f"Pretrain dataset path not found: {self.pretrain_dataset_path}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle existing experiment names by making them unique
        self.experiment_name = self._ensure_unique_experiment_name()
        
        # Initialize experiment
        self.experiment = IDyOMExperiment(
            test_dataset_path=str(self.test_dataset_path),
            pretrain_dataset_path=str(self.pretrain_dataset_path) if self.pretrain_dataset_path else None,
            experiment_history_folder_path=str(self.output_dir),
            experiment_logger_name=self.experiment_name
        )
        
        logger.info(f"IDyOM Analyzer initialized")
        logger.info(f"Test dataset: {self.test_dataset_path}")
        if self.pretrain_dataset_path:
            logger.info(f"Pretrain dataset: {self.pretrain_dataset_path}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _ensure_unique_experiment_name(self) -> str:
        """
        Ensure the experiment name is unique by appending timestamp if needed.
        
        Returns:
            Unique experiment name
        """
        if self.experiment_name is None:
            # If no name provided, let IDyOM generate timestamp-based name
            return None
            
        # Check if experiment folder already exists
        potential_path = self.output_dir / self.experiment_name
        if not potential_path.exists():
            # Name is unique, use as-is
            return self.experiment_name
        
        if self.overwrite:
            # User wants to overwrite existing results
            logger.info(f"Overwriting existing experiment folder '{self.experiment_name}'")
            import shutil
            shutil.rmtree(potential_path)
            return self.experiment_name
        
        # Folder exists, append timestamp to make it unique
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        unique_name = f"{self.experiment_name}_{timestamp}"
        
        logger.info(f"Experiment folder '{self.experiment_name}' already exists, using '{unique_name}' instead")
        return unique_name
    
    def analyze_melody(self,
                      target_viewpoints: List[str] = None,
                      source_viewpoints: Union[str, List[str]] = None,
                      models: str = ':both',
                      k: int = 10,
                      detail: int = 3) -> str:
        """
        Analyze melodies using IDyOM.
        
        Args:
            target_viewpoints: List of target viewpoints (e.g., ['cpitch', 'dur'])
            source_viewpoints: Source viewpoints (':select' or list of viewpoints)
            models: Model type (':stm', ':ltm', ':both')
            k: Cross-validation folds
            detail: Output detail level (1-3)
            
        Returns:
            Path to experiment results folder
        """
        # Set defaults
        if target_viewpoints is None:
            target_viewpoints = ['cpitch']
        if source_viewpoints is None:
            source_viewpoints = ['cpitch']
        
        logger.info(f"Starting melody analysis with target viewpoints: {target_viewpoints}")
        
        # Configure parameters
        self.experiment.set_parameters(
            target_viewpoints=target_viewpoints,
            source_viewpoints=source_viewpoints,
            models=models,
            k=k,
            texture=':melody',
            detail=detail
        )
        
        # Run the experiment
        self.experiment.run()
        
        results_path = self.experiment.logger.this_exp_folder
        logger.info(f"Melody analysis complete. Results saved to: {results_path}")
        return results_path
    
    def analyze_harmony(self,
                       target_viewpoints: List[str] = None,
                       source_viewpoints: Union[str, List[str]] = None,
                       models: str = ':both',
                       k: int = 10,
                       detail: int = 3) -> str:
        """
        Analyze harmonies using IDyOM.
        
        Args:
            target_viewpoints: List of harmonic target viewpoints (e.g., ['pc-chord', 'bass-pc'])
            source_viewpoints: Source viewpoints (':select' or list of viewpoints)
            models: Model type (':stm', ':ltm', ':both')
            k: Cross-validation folds
            detail: Output detail level (1-3)
            
        Returns:
            Path to experiment results folder
        """
        # Set defaults for harmonic analysis
        if target_viewpoints is None:
            target_viewpoints = ['pc-chord']
        if source_viewpoints is None:
            source_viewpoints = ['pc-chord']
        
        logger.info(f"Starting harmony analysis with target viewpoints: {target_viewpoints}")
        
        # Configure parameters
        self.experiment.set_parameters(
            target_viewpoints=target_viewpoints,
            source_viewpoints=source_viewpoints,
            models=models,
            k=k,
            texture=':harmony',
            detail=detail
        )
        
        # Run the experiment
        self.experiment.run()
        
        results_path = self.experiment.logger.this_exp_folder
        logger.info(f"Harmony analysis complete. Results saved to: {results_path}")
        return results_path
    
    def export_results(self, 
                      results_path: str,
                      format: str = 'csv',
                      output_keywords: List[str] = None,
                      melody_names: List[str] = None) -> str:
        """
        Export IDyOM results to different formats.
        
        Args:
            results_path: Path to experiment results folder
            format: Export format ('csv' or 'mat')
            output_keywords: List of output keywords to export
            melody_names: List of specific melodies to export
            
        Returns:
            Path to exported files
        """
        logger.info(f"Exporting results from {results_path} in {format} format")
        
        exporter = Export(
            experiment_folder_path=results_path,
            idyom_output_keywords=output_keywords,
            melody_names=melody_names
        )
        
        if format.lower() == 'csv':
            exporter.export2csv()
        elif format.lower() == 'mat':
            exporter.export2mat()
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'mat'")
        
        export_path = os.path.join(results_path, f'outputs_in_{format}')
        logger.info(f"Export complete: {export_path}")
        return export_path
    
    def get_experiment_info(self, results_path: str) -> ExperimentInfo:
        """
        Get information about a completed experiment.
        
        Args:
            results_path: Path to experiment results folder
            
        Returns:
            ExperimentInfo object with analysis results
        """
        return ExperimentInfo(experiment_folder_path=results_path)


def main():
    """Command-line interface for IDyOM analysis."""
    parser = argparse.ArgumentParser(description='IDyOM Music Analysis Tool')
    parser.add_argument('test_dataset', help='Path to test dataset (MIDI or Kern files)')
    parser.add_argument('--pretrain-dataset', help='Path to pretraining dataset')
    parser.add_argument('--output-dir', default='./idyom_results', help='Output directory')
    parser.add_argument('--experiment-name', help='Name for this experiment')
    parser.add_argument('--texture', choices=['melody', 'harmony'], default='melody', help='Analysis texture')
    parser.add_argument('--target-viewpoints', nargs='+', help='Target viewpoints')
    parser.add_argument('--source-viewpoints', nargs='+', help='Source viewpoints')
    parser.add_argument('--models', choices=[':stm', ':ltm', ':both'], default=':both', help='Model type')
    parser.add_argument('--k', type=int, default=10, help='Cross-validation folds')
    parser.add_argument('--detail', type=int, choices=[1, 2, 3], default=3, help='Output detail level')
    parser.add_argument('--export-format', choices=['csv', 'mat'], help='Export format')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing experiment results')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = IDyOMAnalyzer(
            test_dataset_path=args.test_dataset,
            pretrain_dataset_path=args.pretrain_dataset,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            overwrite=args.overwrite
        )
        
        # Run analysis
        if args.texture == 'melody':
            results_path = analyzer.analyze_melody(
                target_viewpoints=args.target_viewpoints,
                source_viewpoints=args.source_viewpoints,
                models=args.models,
                k=args.k,
                detail=args.detail
            )
        else:  # harmony
            results_path = analyzer.analyze_harmony(
                target_viewpoints=args.target_viewpoints,
                source_viewpoints=args.source_viewpoints,
                models=args.models,
                k=args.k,
                detail=args.detail
            )
        
        # Export results if requested
        if args.export_format:
            analyzer.export_results(results_path, args.export_format)
        
        print(f"Analysis complete! Results saved to: {results_path}")
        
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 