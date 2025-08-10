#!/usr/bin/env python3
"""
Analysis Service

This service handles the analysis of Formula One data to generate insights,
statistics, and performance metrics for drivers, teams, and races.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import statistics

import pandas as pd
import numpy as np
from pydantic import BaseModel

from ..handler import SeasonData, RaceData, DriverData, TeamData

logger = logging.getLogger(__name__)


class AnalysisResult(BaseModel):
    """Container for analysis results."""
    season: int
    analysis_timestamp: datetime
    driver_rankings: List[Dict[str, Any]]
    team_rankings: List[Dict[str, Any]]
    race_insights: List[Dict[str, Any]]
    season_trends: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class AnalysisService:
    """Service for analyzing Formula One data and generating insights."""
    
    def __init__(self):
        """Initialize the analysis service."""
        self.analysis_cache = {}
        
    async def analyze_data(self, season_data: SeasonData) -> AnalysisResult:
        """Analyze Formula One season data and generate insights.
        
        Args:
            season_data: SeasonData object containing all collected data
            
        Returns:
            AnalysisResult object containing analysis results
        """
        try:
            logger.info(f"Starting analysis for season {season_data.season}")
            
            # Analyze driver performance
            driver_rankings = await self._analyze_driver_performance(season_data)
            
            # Analyze team performance
            team_rankings = await self._analyze_team_performance(season_data)
            
            # Analyze race insights
            race_insights = await self._analyze_race_insights(season_data)
            
            # Analyze season trends
            season_trends = await self._analyze_season_trends(season_data)
            
            # Calculate performance metrics
            performance_metrics = await self._calculate_performance_metrics(season_data)
            
            # Create analysis result
            result = AnalysisResult(
                season=season_data.season,
                analysis_timestamp=datetime.now(),
                driver_rankings=driver_rankings,
                team_rankings=team_rankings,
                race_insights=race_insights,
                season_trends=season_trends,
                performance_metrics=performance_metrics
            )
            
            logger.info(f"Analysis completed for season {season_data.season}")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing data: {e}")
            raise
    
    async def _analyze_driver_performance(self, season_data: SeasonData) -> List[Dict[str, Any]]:
        """Analyze driver performance across the season.
        
        Args:
            season_data: Season data to analyze
            
        Returns:
            List of driver performance rankings
        """
        try:
            driver_stats = defaultdict(lambda: {
                'total_points': 0,
                'races_finished': 0,
                'podiums': 0,
                'wins': 0,
                'fastest_laps': 0,
                'positions': [],
                'points_per_race': 0
            })
            
            # Aggregate driver statistics across all races
            for race in season_data.races:
                for result in race.results:
                    driver_id = result.driver_id
                    stats = driver_stats[driver_id]
                    
                    # Basic stats
                    stats['total_points'] += result.points or 0
                    if result.position and result.position <= 20:  # Finished race
                        stats['races_finished'] += 1
                        stats['positions'].append(result.position)
                    
                    # Achievements
                    if result.position == 1:
                        stats['wins'] += 1
                    if result.position <= 3:
                        stats['podiums'] += 1
                    
                    # Fastest lap
                    if race.fastest_lap and race.fastest_lap.driver_id == driver_id:
                        stats['fastest_laps'] += 1
            
            # Calculate additional metrics
            for driver_id, stats in driver_stats.items():
                if stats['races_finished'] > 0:
                    stats['points_per_race'] = stats['total_points'] / stats['races_finished']
                    stats['average_position'] = statistics.mean(stats['positions']) if stats['positions'] else None
                    stats['best_position'] = min(stats['positions']) if stats['positions'] else None
                    stats['worst_position'] = max(stats['positions']) if stats['positions'] else None
            
            # Sort by total points
            sorted_drivers = sorted(
                driver_stats.items(),
                key=lambda x: x[1]['total_points'],
                reverse=True
            )
            
            # Format results
            rankings = []
            for rank, (driver_id, stats) in enumerate(sorted_drivers, 1):
                # Find driver info from first race
                driver_info = None
                for race in season_data.races:
                    for result in race.results:
                        if result.driver_id == driver_id:
                            driver_info = result
                            break
                    if driver_info:
                        break
                
                if driver_info:
                    rankings.append({
                        'rank': rank,
                        'driver_id': driver_id,
                        'driver_name': f"{driver_info.first_name} {driver_info.last_name}",
                        'driver_code': driver_info.driver_code,
                        'team': driver_info.team,
                        'total_points': stats['total_points'],
                        'races_finished': stats['races_finished'],
                        'podiums': stats['podiums'],
                        'wins': stats['wins'],
                        'fastest_laps': stats['fastest_laps'],
                        'average_position': stats['average_position'],
                        'best_position': stats['best_position'],
                        'worst_position': stats['worst_position'],
                        'points_per_race': round(stats['points_per_race'], 2)
                    })
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error analyzing driver performance: {e}")
            return []
    
    async def _analyze_team_performance(self, season_data: SeasonData) -> List[Dict[str, Any]]:
        """Analyze team performance across the season.
        
        Args:
            season_data: Season data to analyze
            
        Returns:
            List of team performance rankings
        """
        try:
            team_stats = defaultdict(lambda: {
                'total_points': 0,
                'races_finished': 0,
                'podiums': 0,
                'wins': 0,
                'fastest_laps': 0,
                'drivers': set(),
                'best_finish': float('inf'),
                'worst_finish': 0
            })
            
            # Aggregate team statistics across all races
            for race in season_data.races:
                for result in race.results:
                    team_name = result.team
                    stats = team_stats[team_name]
                    
                    # Basic stats
                    stats['total_points'] += result.points or 0
                    stats['drivers'].add(result.driver_id)
                    
                    if result.position and result.position <= 20:
                        stats['races_finished'] += 1
                        stats['best_finish'] = min(stats['best_finish'], result.position)
                        stats['worst_finish'] = max(stats['worst_finish'], result.position)
                    
                    # Achievements
                    if result.position == 1:
                        stats['wins'] += 1
                    if result.position <= 3:
                        stats['podiums'] += 1
                    
                    # Fastest lap
                    if race.fastest_lap and race.fastest_lap.team == team_name:
                        stats['fastest_laps'] += 1
            
            # Calculate additional metrics
            for team_name, stats in team_stats.items():
                stats['driver_count'] = len(stats['drivers'])
                stats['points_per_race'] = stats['total_points'] / stats['races_finished'] if stats['races_finished'] > 0 else 0
                stats['best_finish'] = stats['best_finish'] if stats['best_finish'] != float('inf') else None
            
            # Sort by total points
            sorted_teams = sorted(
                team_stats.items(),
                key=lambda x: x[1]['total_points'],
                reverse=True
            )
            
            # Format results
            rankings = []
            for rank, (team_name, stats) in enumerate(sorted_teams, 1):
                rankings.append({
                    'rank': rank,
                    'team_name': team_name,
                    'total_points': stats['total_points'],
                    'races_finished': stats['races_finished'],
                    'podiums': stats['podiums'],
                    'wins': stats['wins'],
                    'fastest_laps': stats['fastest_laps'],
                    'driver_count': stats['driver_count'],
                    'best_finish': stats['best_finish'],
                    'worst_finish': stats['worst_finish'],
                    'points_per_race': round(stats['points_per_race'], 2)
                })
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error analyzing team performance: {e}")
            return []
    
    async def _analyze_race_insights(self, season_data: SeasonData) -> List[Dict[str, Any]]:
        """Analyze individual races for insights.
        
        Args:
            season_data: Season data to analyze
            
        Returns:
            List of race insights
        """
        try:
            insights = []
            
            for race in season_data.races:
                if not race.results:
                    continue
                
                # Calculate race statistics
                total_points = sum(result.points or 0 for result in race.results)
                finishers = [r for r in race.results if r.position and r.position <= 20]
                dnf_count = len(race.results) - len(finishers)
                
                # Find interesting patterns
                insights.append({
                    'race_id': race.race_id,
                    'race_name': race.race_name,
                    'round': race.round_number,
                    'date': race.date,
                    'total_points_awarded': total_points,
                    'finishers': len(finishers),
                    'dnf_count': dnf_count,
                    'dnf_rate': round(dnf_count / len(race.results) * 100, 1),
                    'fastest_lap_driver': race.fastest_lap.first_name + " " + race.fastest_lap.last_name if race.fastest_lap else None,
                    'winner': next((r.first_name + " " + r.last_name for r in race.results if r.position == 1), None)
                })
            
            return insights
            
        except Exception as e:
            logger.error(f"Error analyzing race insights: {e}")
            return []
    
    async def _analyze_season_trends(self, season_data: SeasonData) -> Dict[str, Any]:
        """Analyze trends across the season.
        
        Args:
            season_data: Season data to analyze
            
        Returns:
            Dictionary containing season trends
        """
        try:
            trends = {
                'total_races': len(season_data.races),
                'season_duration_days': None,
                'points_distribution': {},
                'position_volatility': {}
            }
            
            if len(season_data.races) >= 2:
                # Calculate season duration
                first_race = min(race.date for race in season_data.races)
                last_race = max(race.date for race in season_data.races)
                trends['season_duration_days'] = (last_race - first_race).days
                
                # Analyze points distribution over time
                race_points = []
                for race in season_data.races:
                    total_points = sum(result.points or 0 for result in race.results)
                    race_points.append({
                        'round': race.round_number,
                        'points': total_points,
                        'date': race.date
                    })
                
                trends['points_distribution'] = {
                    'total_points': sum(r['points'] for r in race_points),
                    'average_points_per_race': round(statistics.mean(r['points'] for r in race_points), 2),
                    'highest_points_race': max(race_points, key=lambda x: x['points']),
                    'lowest_points_race': min(race_points, key=lambda x: x['points'])
                }
            
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing season trends: {e}")
            return {}
    
    async def _calculate_performance_metrics(self, season_data: SeasonData) -> Dict[str, Any]:
        """Calculate overall performance metrics for the season.
        
        Args:
            season_data: Season data to analyze
            
        Returns:
            Dictionary containing performance metrics
        """
        try:
            metrics = {
                'data_quality': {},
                'coverage': {},
                'statistics': {}
            }
            
            # Data quality metrics
            total_races = len(season_data.races)
            races_with_results = sum(1 for race in season_data.races if race.results)
            races_with_fastest_lap = sum(1 for race in season_data.races if race.fastest_lap)
            
            metrics['data_quality'] = {
                'total_races': total_races,
                'races_with_results': races_with_results,
                'races_with_fastest_lap': races_with_fastest_lap,
                'completeness_rate': round(races_with_results / total_races * 100, 1) if total_races > 0 else 0
            }
            
            # Coverage metrics
            if season_data.drivers_championship:
                metrics['coverage']['drivers_championship'] = len(season_data.drivers_championship)
            if season_data.constructors_championship:
                metrics['coverage']['constructors_championship'] = len(season_data.constructors_championship)
            
            # Statistical metrics
            all_positions = []
            all_points = []
            
            for race in season_data.races:
                for result in race.results:
                    if result.position:
                        all_positions.append(result.position)
                    if result.points:
                        all_points.append(result.points)
            
            if all_positions:
                metrics['statistics']['position_stats'] = {
                    'mean': round(statistics.mean(all_positions), 2),
                    'median': statistics.median(all_positions),
                    'std_dev': round(statistics.stdev(all_positions), 2) if len(all_positions) > 1 else None
                }
            
            if all_points:
                metrics['statistics']['points_stats'] = {
                    'mean': round(statistics.mean(all_points), 2),
                    'median': statistics.median(all_points),
                    'std_dev': round(statistics.stdev(all_points), 2) if len(all_points) > 1 else None
                }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    async def cleanup(self):
        """Clean up resources."""
        try:
            self.analysis_cache.clear()
            logger.info("Analysis service cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during analysis service cleanup: {e}")
