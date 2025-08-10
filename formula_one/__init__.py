"""
Formula One Data Analysis Module

A comprehensive Python module for analyzing and presenting statistics on Formula 1 
teams and drivers performance.
"""

from .main import FormulaOneApp, main
from .handler import (
    FormulaOneHandler,
    DriverData,
    TeamData,
    RaceData,
    SeasonData
)
from .services import (
    DataCollectionService,
    AnalysisService,
    ReportingService
)

# Version information
__version__ = "0.1.0"
__author__ = "Edwin Gelebart"
__email__ = "edwin.gelebart@gmail.com"

# Main exports
__all__ = [
    # Main application
    "FormulaOneApp",
    "main",
    
    # Core handler and data models
    "FormulaOneHandler",
    "DriverData",
    "TeamData", 
    "RaceData",
    "SeasonData",
    
    # Services
    "DataCollectionService",
    "AnalysisService",
    "ReportingService",
]