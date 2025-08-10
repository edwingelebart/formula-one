#!/usr/bin/env python3
"""
Reporting Service

This service handles the generation and export of Formula One analysis reports
to various formats including Google Sheets, CSV, and JSON.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import csv

import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

from ..handler import SeasonData
from .analysis import AnalysisResult

logger = logging.getLogger(__name__)


class ReportFormat:
    """Supported report formats."""
    JSON = "json"
    CSV = "csv"
    GOOGLE_SHEETS = "google_sheets"
    EXCEL = "excel"


class ReportingService:
    """Service for generating and exporting Formula One analysis reports."""
    
    def __init__(self, google_credentials_path: Optional[Path] = None):
        """Initialize the reporting service.
        
        Args:
            google_credentials_path: Path to Google service account credentials
        """
        self.google_credentials_path = google_credentials_path
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        self._google_client = None
        
    async def generate_reports(self, analysis_result: AnalysisResult, 
                             formats: Optional[List[str]] = None) -> Dict[str, str]:
        """Generate reports in the specified formats.
        
        Args:
            analysis_result: Analysis results to report on
            formats: List of report formats to generate (default: all formats)
            
        Returns:
            Dictionary mapping format to report file path
        """
        try:
            if formats is None:
                formats = [ReportFormat.JSON, ReportFormat.CSV, ReportFormat.GOOGLE_SHEETS]
            
            logger.info(f"Generating reports in formats: {formats}")
            
            report_paths = {}
            
            # Generate timestamp for report naming
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            season = analysis_result.season
            
            # Generate JSON report
            if ReportFormat.JSON in formats:
                json_path = await self._generate_json_report(analysis_result, timestamp, season)
                report_paths[ReportFormat.JSON] = str(json_path)
            
            # Generate CSV reports
            if ReportFormat.CSV in formats:
                csv_paths = await self._generate_csv_reports(analysis_result, timestamp, season)
                report_paths[ReportFormat.CSV] = csv_paths
            
            # Generate Google Sheets report
            if ReportFormat.GOOGLE_SHEETS in formats:
                sheets_url = await self._generate_google_sheets_report(analysis_result, season)
                report_paths[ReportFormat.GOOGLE_SHEETS] = sheets_url
            
            # Generate Excel report
            if ReportFormat.EXCEL in formats:
                excel_path = await self._generate_excel_report(analysis_result, timestamp, season)
                report_paths[ReportFormat.EXCEL] = str(excel_path)
            
            logger.info(f"Reports generated successfully: {list(report_paths.keys())}")
            return report_paths
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
            raise
    
    async def _generate_json_report(self, analysis_result: AnalysisResult, 
                                  timestamp: str, season: int) -> Path:
        """Generate a comprehensive JSON report.
        
        Args:
            analysis_result: Analysis results to report on
            timestamp: Timestamp string for file naming
            season: F1 season year
            
        Returns:
            Path to the generated JSON file
        """
        try:
            # Convert to dict for JSON serialization
            report_data = {
                'metadata': {
                    'generated_at': analysis_result.analysis_timestamp.isoformat(),
                    'season': analysis_result.season,
                    'report_version': '1.0'
                },
                'driver_rankings': analysis_result.driver_rankings,
                'team_rankings': analysis_result.team_rankings,
                'race_insights': analysis_result.race_insights,
                'season_trends': analysis_result.season_trends,
                'performance_metrics': analysis_result.performance_metrics
            }
            
            # Create filename
            filename = f"f1_season_{season}_analysis_{timestamp}.json"
            file_path = self.reports_dir / filename
            
            # Write JSON file
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"JSON report generated: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {e}")
            raise
    
    async def _generate_csv_reports(self, analysis_result: AnalysisResult, 
                                  timestamp: str, season: int) -> Dict[str, str]:
        """Generate CSV reports for different data categories.
        
        Args:
            analysis_result: Analysis results to report on
            timestamp: Timestamp string for file naming
            season: F1 season year
            
        Returns:
            Dictionary mapping report type to file path
        """
        try:
            csv_paths = {}
            
            # Driver rankings CSV
            if analysis_result.driver_rankings:
                driver_filename = f"f1_season_{season}_driver_rankings_{timestamp}.csv"
                driver_path = self.reports_dir / driver_filename
                
                with open(driver_path, 'w', newline='', encoding='utf-8') as f:
                    if analysis_result.driver_rankings:
                        writer = csv.DictWriter(f, fieldnames=analysis_result.driver_rankings[0].keys())
                        writer.writeheader()
                        writer.writerows(analysis_result.driver_rankings)
                
                csv_paths['driver_rankings'] = str(driver_path)
            
            # Team rankings CSV
            if analysis_result.team_rankings:
                team_filename = f"f1_season_{season}_team_rankings_{timestamp}.csv"
                team_path = self.reports_dir / team_filename
                
                with open(team_path, 'w', newline='', encoding='utf-8') as f:
                    if analysis_result.team_rankings:
                        writer = csv.DictWriter(f, fieldnames=analysis_result.team_rankings[0].keys())
                        writer.writeheader()
                        writer.writerows(analysis_result.team_rankings)
                
                csv_paths['team_rankings'] = str(team_path)
            
            # Race insights CSV
            if analysis_result.race_insights:
                race_filename = f"f1_season_{season}_race_insights_{timestamp}.csv"
                race_path = self.reports_dir / race_filename
                
                with open(race_path, 'w', newline='', encoding='utf-8') as f:
                    if analysis_result.race_insights:
                        writer = csv.DictWriter(f, fieldnames=analysis_result.race_insights[0].keys())
                        writer.writeheader()
                        writer.writerows(analysis_result.race_insights)
                
                csv_paths['race_insights'] = str(race_path)
            
            logger.info(f"CSV reports generated: {list(csv_paths.keys())}")
            return csv_paths
            
        except Exception as e:
            logger.error(f"Error generating CSV reports: {e}")
            raise
    
    async def _generate_google_sheets_report(self, analysis_result: AnalysisResult, 
                                           season: int) -> str:
        """Generate a Google Sheets report.
        
        Args:
            analysis_result: Analysis results to report on
            season: F1 season year
            
        Returns:
            URL to the generated Google Sheets document
        """
        try:
            if not self.google_credentials_path or not self.google_credentials_path.exists():
                logger.warning("Google credentials not found, skipping Google Sheets report")
                return "Google Sheets report not generated - credentials missing"
            
            # Initialize Google Sheets client
            await self._initialize_google_client()
            
            if not self._google_client:
                return "Google Sheets report not generated - client initialization failed"
            
            # Create new spreadsheet
            spreadsheet_name = f"F1 Season {season} Analysis - {datetime.now().strftime('%Y-%m-%d')}"
            spreadsheet = self._google_client.create(spreadsheet_name)
            
            # Add driver rankings sheet
            if analysis_result.driver_rankings:
                driver_sheet = spreadsheet.add_worksheet(title="Driver Rankings", rows=100, cols=20)
                await self._populate_google_sheet(driver_sheet, analysis_result.driver_rankings)
            
            # Add team rankings sheet
            if analysis_result.team_rankings:
                team_sheet = spreadsheet.add_worksheet(title="Team Rankings", rows=100, cols=20)
                await self._populate_google_sheet(team_sheet, analysis_result.team_rankings)
            
            # Add race insights sheet
            if analysis_result.race_insights:
                race_sheet = spreadsheet.add_worksheet(title="Race Insights", rows=100, cols=20)
                await self._populate_google_sheet(race_sheet, analysis_result.race_insights)
            
            # Add summary sheet
            summary_sheet = spreadsheet.add_worksheet(title="Season Summary", rows=50, cols=20)
            await self._populate_summary_sheet(summary_sheet, analysis_result)
            
            # Remove default sheet
            spreadsheet.del_worksheet(spreadsheet.sheet1)
            
            # Make spreadsheet accessible
            spreadsheet.share('', perm_type='anyone', role='reader')
            
            logger.info(f"Google Sheets report generated: {spreadsheet.url}")
            return spreadsheet.url
            
        except Exception as e:
            logger.error(f"Error generating Google Sheets report: {e}")
            return f"Google Sheets report failed: {str(e)}"
    
    async def _generate_excel_report(self, analysis_result: AnalysisResult, 
                                   timestamp: str, season: int) -> Path:
        """Generate an Excel report with multiple sheets.
        
        Args:
            analysis_result: Analysis results to report on
            timestamp: Timestamp string for file naming
            season: F1 season year
            
        Returns:
            Path to the generated Excel file
        """
        try:
            filename = f"f1_season_{season}_analysis_{timestamp}.xlsx"
            file_path = self.reports_dir / filename
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Driver rankings sheet
                if analysis_result.driver_rankings:
                    df_drivers = pd.DataFrame(analysis_result.driver_rankings)
                    df_drivers.to_excel(writer, sheet_name='Driver Rankings', index=False)
                
                # Team rankings sheet
                if analysis_result.team_rankings:
                    df_teams = pd.DataFrame(analysis_result.team_rankings)
                    df_teams.to_excel(writer, sheet_name='Team Rankings', index=False)
                
                # Race insights sheet
                if analysis_result.race_insights:
                    df_races = pd.DataFrame(analysis_result.race_insights)
                    df_races.to_excel(writer, sheet_name='Race Insights', index=False)
                
                # Season summary sheet
                summary_data = {
                    'Metric': ['Season', 'Total Races', 'Analysis Date'],
                    'Value': [
                        analysis_result.season,
                        len(analysis_result.race_insights),
                        analysis_result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                    ]
                }
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Season Summary', index=False)
            
            logger.info(f"Excel report generated: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error generating Excel report: {e}")
            raise
    
    async def _initialize_google_client(self):
        """Initialize the Google Sheets client."""
        try:
            if self._google_client:
                return
            
            if not self.google_credentials_path:
                logger.warning("No Google credentials path provided")
                return
            
            # Load credentials
            scope = ['https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive']
            
            credentials = Credentials.from_service_account_file(
                str(self.google_credentials_path), scopes=scope
            )
            
            self._google_client = gspread.authorize(credentials)
            logger.info("Google Sheets client initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Google Sheets client: {e}")
            self._google_client = None
    
    async def _populate_google_sheet(self, worksheet, data: List[Dict[str, Any]]):
        """Populate a Google Sheets worksheet with data.
        
        Args:
            worksheet: Google Sheets worksheet object
            data: List of dictionaries containing data
        """
        try:
            if not data:
                return
            
            # Prepare headers and data
            headers = list(data[0].keys())
            rows = [headers]
            
            for item in data:
                row = []
                for header in headers:
                    value = item.get(header, '')
                    # Convert datetime objects to strings
                    if isinstance(value, datetime):
                        value = value.strftime('%Y-%m-%d %H:%M:%S')
                    row.append(str(value))
                rows.append(row)
            
            # Update worksheet
            worksheet.update(rows)
            
        except Exception as e:
            logger.error(f"Error populating Google Sheet: {e}")
    
    async def _populate_summary_sheet(self, worksheet, analysis_result: AnalysisResult):
        """Populate the summary sheet with season overview.
        
        Args:
            worksheet: Google Sheets worksheet object
            analysis_result: Analysis results to summarize
        """
        try:
            summary_data = [
                ['Season', analysis_result.season],
                ['Analysis Date', analysis_result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')],
                ['Total Races', len(analysis_result.race_insights)],
                ['Total Drivers', len(analysis_result.driver_rankings)],
                ['Total Teams', len(analysis_result.team_rankings)],
                [''],
                ['Top 3 Drivers'],
                ['Rank', 'Driver', 'Points', 'Wins'],
            ]
            
            # Add top 3 drivers
            for i, driver in enumerate(analysis_result.driver_rankings[:3], 1):
                summary_data.append([
                    i,
                    driver['driver_name'],
                    driver['total_points'],
                    driver['wins']
                ])
            
            summary_data.extend([
                [''],
                ['Top 3 Teams'],
                ['Rank', 'Team', 'Points', 'Wins'],
            ])
            
            # Add top 3 teams
            for i, team in enumerate(analysis_result.team_rankings[:3], 1):
                summary_data.append([
                    i,
                    team['team_name'],
                    team['total_points'],
                    team['wins']
                ])
            
            # Update worksheet
            worksheet.update(summary_data)
            
        except Exception as e:
            logger.error(f"Error populating summary sheet: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            # Close Google Sheets client if it exists
            if self._google_client:
                # Note: gspread doesn't have a close method, but we can clear the reference
                self._google_client = None
            
            logger.info("Reporting service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during reporting service cleanup: {e}")
