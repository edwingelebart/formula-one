#!/usr/bin/env python3
"""
Google Sheets Client Service

This service provides a simplified interface for Google Sheets operations
using gspread, including authentication, CRUD operations, and data management.
"""

import logging
from pathlib import Path
from typing import Any


import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

logger = logging.getLogger(__name__)


class GSpreadClient:
    """Google Sheets client service with essential operations."""
    
    def __init__(self, credentials_path: Path):
        """Initialize the Google Sheets client.
        
        Args:
            credentials_path: Path to Google service account credentials JSON file
        """        
        # Define required scopes
        scope = [
                'https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive',
                'https://www.googleapis.com/auth/spreadsheets'
            ]

        # Load credentials
        credentials = Credentials.from_service_account_file(
            str(credentials_path), scopes=scope
        )

        self.client = gspread.authorize(credentials)
        
    
    async def create_spreadsheet(self, title: str, share_email: str | None) -> str | None:
        """Create a new Google Spreadsheet.
        
        Args:
            title: Title of the spreadsheet
            share_email: Optional email to share the spreadsheet with
            
        Returns:
            Spreadsheet URL if successful, None otherwise
        """
        try:            
            # Create spreadsheet
            spreadsheet = self.client.create(title)
            
            # Share if email provided
            if share_email:
                spreadsheet.share(share_email, perm_type='user', role='writer')
            
            # Make publicly readable
            spreadsheet.share('', perm_type='anyone', role='reader')
            
            logger.info(f"Created spreadsheet: {title} - {spreadsheet.url}")
            return spreadsheet.url
            
        except Exception as e:
            logger.error(f"Error creating spreadsheet: {e}")
            return None
    
    async def open_spreadsheet(self, spreadsheet_id: str) -> gspread.Spreadsheet | None:
        """Open an existing spreadsheet by ID.
        
        Args:
            spreadsheet_id: The spreadsheet ID from the URL
            
        Returns:
            Spreadsheet object if successful, None otherwise
        """
        try:            
            spreadsheet = self.client.open_by_key(spreadsheet_id)
            logger.info(f"Opened spreadsheet: {spreadsheet.title}")
            return spreadsheet
            
        except Exception as e:
            logger.error(f"Error opening spreadsheet: {e}")
            return None
    
    async def create_worksheet(
        self, 
        spreadsheet: gspread.Spreadsheet, 
        title: str, 
        rows: int = 100, 
        cols: int = 20
    ) -> gspread.Worksheet | None:
        """Create a new worksheet in a spreadsheet.
        
        Args:
            spreadsheet: The spreadsheet object
            title: Title of the worksheet
            rows: Number of rows
            cols: Number of columns
            
        Returns:
            Worksheet object if successful, None otherwise
        """
        try:
            worksheet = spreadsheet.add_worksheet(title=title, rows=rows, cols=cols)
            logger.info(f"Created worksheet: {title}")
            return worksheet
            
        except Exception as e:
            logger.error(f"Error creating worksheet: {e}")
            return None
    
    async def write_data(self, worksheet: gspread.Worksheet, data: list[Any]) -> bool:
        """Write data to a worksheet.
        
        Args:
            worksheet: The worksheet to write to
            data: 2D list of data (rows x columns)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            worksheet.update(data)
            logger.info(f"Wrote {len(data)} rows to worksheet: {worksheet.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error writing data to worksheet: {e}")
            return False
    
    async def write_dataframe(self, worksheet: gspread.Worksheet, df: pd.DataFrame) -> bool:
        """Write a pandas DataFrame to a worksheet.
        
        Args:
            worksheet: The worksheet to write to
            df: Pandas DataFrame to write
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert DataFrame to list of lists (including headers)
            data = [df.columns.tolist()] + df.values.tolist()
            return await self.write_data(worksheet, data)
            
        except Exception as e:
            logger.error(f"Error writing DataFrame to worksheet: {e}")
            return False
    
    async def read_data(
        self, 
        worksheet: gspread.Worksheet, 
        range_name: str | None = None
    ) -> list[[Any]] | None:
        """Read data from a worksheet.
        
        Args:
            worksheet: The worksheet to read from
            range_name: Optional range (e.g., 'A1:D10'), defaults to all data
            
        Returns:
            List of lists containing the data, None if error
        """
        try:
            if range_name:
                data = worksheet.get(range_name)
            else:
                data = worksheet.get_all_values()
            
            logger.info(f"Read {len(data)} rows from worksheet: {worksheet.title}")
            return data
            
        except Exception as e:
            logger.error(f"Error reading data from worksheet: {e}")
            return None
    
    async def read_dataframe(
        self, 
        worksheet: gspread.Worksheet, 
        range_name: str | None = None
    ) -> pd.DataFrame | None:
        """Read data from a worksheet as a pandas DataFrame.
        
        Args:
            worksheet: The worksheet to read from
            range_name: Optional range to read
            
        Returns:
            Pandas DataFrame if successful, None otherwise
        """
        try:
            data = await self.read_data(worksheet, range_name)
            if not data:
                return None
            
            # Convert to DataFrame
            if len(data) > 1:
                df = pd.DataFrame(data[1:], columns=data[0])
            else:
                df = pd.DataFrame(columns=data[0] if data else [])
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading DataFrame from worksheet: {e}")
            return None
    
    async def update_cell(
        self, 
        worksheet: gspread.Worksheet, 
        row: int, 
        col: int, 
        value: Any
    ) -> bool:
        """Update a single cell in a worksheet.
        
        Args:
            worksheet: The worksheet to update
            row: Row number (1-indexed)
            col: Column number (1-indexed)
            value: Value to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            worksheet.update_cell(row, col, value)
            logger.debug(f"Updated cell ({row}, {col}) with value: {value}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating cell ({row}, {col}): {e}")
            return False
    
    async def clear_worksheet(self, worksheet: gspread.Worksheet) -> bool:
        """Clear all data from a worksheet.
        
        Args:
            worksheet: The worksheet to clear
            
        Returns:
            True if successful, False otherwise
        """
        try:
            worksheet.clear()
            logger.info(f"Cleared worksheet: {worksheet.title}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing worksheet: {e}")
            return False
    
    async def delete_worksheet(self, worksheet: gspread.Worksheet) -> bool:
        """Delete a worksheet.
        
        Args:
            worksheet: The worksheet to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            spreadsheet = worksheet.spreadsheet
            worksheet_title = worksheet.title
            spreadsheet.del_worksheet(worksheet)
            logger.info(f"Deleted worksheet: {worksheet_title}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting worksheet: {e}")
            return False
