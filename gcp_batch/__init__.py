"""
GCP Batch Training Module

Provides tools for launching and managing AI Toolkit training jobs on Google Cloud Batch.
"""

from .gcp_batch_launcher import GCPBatchLauncher

__version__ = "1.0.0"
__all__ = ['GCPBatchLauncher']
