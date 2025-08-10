#!/usr/bin/env python3
"""
Example Usage of Formula One Analysis System

This script demonstrates how to use the Formula One data analysis system
to collect, analyze, and report on Formula One season data.
"""

import asyncio
import logging
from pathlib import Path

from formula_one.main import FormulaOneApp


async def main():
    """Example usage of the Formula One analysis system."""
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize the application
        app = FormulaOneApp()
        
        # Example 1: Analyze a specific season
        logger.info("Example 1: Analyzing Formula One 2024 season")
        await app.run(season=2024)
        
        # Example 2: Analyze specific rounds from a season
        logger.info("Example 2: Analyzing specific rounds from 2023 season")
        await app.run(season=2023, rounds=[1, 2, 3])  # First 3 races
        
        # Example 3: Custom configuration
        logger.info("Example 3: Using custom configuration")
        custom_app = FormulaOneApp(config_path=Path("custom_config.yaml"))
        await custom_app.run(season=2024)
        await custom_app.cleanup()
        
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
    
    finally:
        # Clean up
        await app.cleanup()


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
