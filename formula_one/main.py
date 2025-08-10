#!/usr/bin/env python3
"""
Formula One Data Analysis - Main Entry Point

This module serves as the main entry point for the Formula One data analysis application.
It coordinates the data collection, processing, and analysis workflows.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from .handler import FormulaOneHandler
from .services import DataCollectionService, AnalysisService, ReportingService, GSpreadClient


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('formula_one.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FormulaOneApp:
    """Main application class for Formula One data analysis."""
    
    def __init__(self):
        """Initialize the Formula One application.
        
        Args:
            config_path: Optional path to configuration file
        """
        gspread_client = GSpreadClient()
        self.handler = FormulaOneHandler(
            gspread_client=gspread_client
        )
        
    async def run(self):
        """Run the main application workflow.
        
        Args:
            season: F1 season year to analyze
            rounds: Optional list of specific race rounds to analyze
        """
        try:
            logger.info(f"Starting Formula One analysis for season {season}")
            
            # Initialize services
            await self.handler.initialize()
            
            # Collect data
            logger.info("Collecting Formula One data...")
            data = await self.data_service.collect_season_data(season, rounds)
            
            # Analyze data
            logger.info("Analyzing collected data...")
            analysis_results = await self.analysis_service.analyze_data(data)
            
            # Generate reports
            logger.info("Generating reports...")
            await self.reporting_service.generate_reports(analysis_results)
            
            logger.info("Formula One analysis completed successfully!")
            
        except Exception as e:
            logger.error(f"Error running Formula One analysis: {e}")
            raise
    


async def main():
    """Main entry point for the application."""
    app = FormulaOneApp()
    
    # Run the application
    await app.run()



if __name__ == "__main__":
    
    asyncio.run(main())
