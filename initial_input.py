"""
f1_elo_module.py

A self-contained module to build and maintain ELO ratings for Formula 1 drivers and constructors.

Features
- Object-oriented design
- Pydantic models for data validation
- FastF1 (via Jolpica) data fetching helper
- EloCalculator class (driver & constructor)
- Storage backend with optional Google Sheets (gspread) or local CSV fallback
- Fully documented and ready to be integrated into a pipeline or run manually after each race

Usage (minimal):

    from f1_elo_module import F1EloManager

    mgr = F1EloManager(cache_dir='f1_cache')
    mgr.initialize_ratings_from_ergast(year=2023)  # optional - bootstrap
    results = mgr.fetch_race_results(2023, 7)
    mgr.update_ratings_for_race(results)
    mgr.save_ratings('ratings.csv')

Dependencies (declare with uv):
- fastf1
- pydantic
- pandas
- gspread (optional, for Google Sheets storage)
- oauth2client (optional, for Google Sheets auth)

See the bottom of this file for a suggested pyproject.toml and uv commands.

"""
from __future__ import annotations

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
import pandas as pd
import fastf1
import fastf1.ergast
import pathlib
import logging
import math
import csv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# -----------------------------
# Pydantic models
# -----------------------------

class DriverModel(BaseModel):
    driver_id: str = Field(..., description="Ergast driverId or unique code")
    given_name: Optional[str]
    family_name: Optional[str]
    code: Optional[str]
    nationality: Optional[str]
    elo: float = Field(1500.0, description="Current ELO rating")
    last_updated: Optional[datetime]

    @property
    def full_name(self) -> str:
        if self.given_name and self.family_name:
            return f"{self.given_name} {self.family_name}"
        return self.driver_id


class ConstructorModel(BaseModel):
    constructor_id: str = Field(..., description="Ergast constructorId or unique code")
    name: Optional[str]
    nationality: Optional[str]
    elo: float = Field(1500.0)
    last_updated: Optional[datetime]


class RaceResultModel(BaseModel):
    year: int
    round: int
    race_name: Optional[str]
    date: Optional[datetime]
    results: List[Dict[str, Any]]  # raw results rows from ergast (dicts)

    @validator('results')
    def must_have_results(cls, v):
        if not v or len(v) == 0:
            raise ValueError('results must be a non-empty list')
        return v


# -----------------------------
# Helpers: Ergast / FastF1 wrapper
# -----------------------------

class FastF1Fetcher:
    """Wraps FastF1 fetches and Ergast (Jolpica) endpoint.

    Notes:
    - By default this sets FastF1 cache and switches the Ergast base URL to Jolpica.
    - Methods return pandas DataFrames where appropriate.
    """

    def __init__(self, cache_dir: str = 'f1_cache'):
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(str(self.cache_dir))
        # switch to Jolpica (Ergast successor)
        try:
            fastf1.ergast.interface.BASE_URL = "https://api.jolpi.ca/ergast/f1"
        except Exception:
            logger.warning('Could not set Ergast BASE_URL; continuing with default')

    def get_race_results_df(self, year: int, round_num: int) -> pd.DataFrame:
        """Fetch race results for a given year and round using Ergast-compatible API.

        Returns a DataFrame where each row is a driver result.
        """
        ergast = fastf1.ergast.Ergast()
        response = ergast.get_race_results(year, round_num)
        # `.content` is a list with a single list of driver result dicts
        try:
            results = response.content[0]
        except Exception as e:
            logger.error('Could not parse ergast response: %s', e)
            raise

        # Normalize to DataFrame. Keep nested Driver and Constructor as columns.
        df = pd.json_normalize(results)
        # Ensure position is numeric where possible
        if 'position' in df.columns:
            df['position'] = pd.to_numeric(df['position'], errors='coerce')
        return df

    def get_session(self, year: int, event: str, session: str):
        """Convenience wrapper for `fastf1.get_session` to load telemetry if needed.

        event can be race name or round number; fastf1 accepts both in many cases.
        session is one of 'FP1','FP2','FP3','Q','R'.
        """
        s = fastf1.get_session(year, event, session)
        s.load()
        return s


# -----------------------------
# ELO calculation
# -----------------------------

class EloCalculator:
    """Elo calculator that supports multi-player events by using pairwise expectation.

    Strategy used:
    - For an N-driver race, treat each driver's finishing position as "winning" against
      lower-placed drivers and "losing" to higher-placed drivers.
    - For each pair (A,B), compute expected score and result (1/0/0.5), sum across pairs,
      normalize by number of pairs to get final score for A in [0,1].
    - Update ELO with K factor applied to that score.

    This is a simple, explainable method. It can be refined (e.g., weighting by gap, pit stops,
    or race importance).
    """

    def __init__(self, k: float = 20.0):
        self.k = float(k)

    @staticmethod
    def expected(a: float, b: float) -> float:
        """Standard Elo expected score for player with rating a vs b."""
        return 1.0 / (1.0 + 10 ** ((b - a) / 400.0))

    def compute_pairwise_scores(self, ratings: Dict[str, float], order: List[str]) -> Dict[str, float]:
        """Given a list `order` of entity IDs in finishing order (best -> worst),
        compute normalized scores in [0,1] for each entity, by comparing pairwise.
        """
        n = len(order)
        if n <= 1:
            return {order[0]: 0.5} if n == 1 else {}

        scores: Dict[str, float] = {eid: 0.0 for eid in order}
        pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                a = order[i]
                b = order[j]
                ra = ratings.get(a, 1500.0)
                rb = ratings.get(b, 1500.0)
                exp_a = self.expected(ra, rb)
                exp_b = 1.0 - exp_a
                # result: a beat b (1), b lost (0)
                scores[a] += 1.0  # actual score for a vs b
                scores[b] += 0.0
                # but we will later compare to expected, so store expected separately
                # We'll compute final delta by (actual_sum - expected_sum)
                pairs += 1

        # Now compute expected sums
        expected_sums: Dict[str, float] = {eid: 0.0 for eid in order}
        for i in range(n):
            for j in range(i + 1, n):
                a = order[i]
                b = order[j]
                ra = ratings.get(a, 1500.0)
                rb = ratings.get(b, 1500.0)
                exp_a = self.expected(ra, rb)
                expected_sums[a] += exp_a
                expected_sums[b] += (1.0 - exp_a)

        # Normalize: each player's actual score is number of wins vs lower-placed drivers
        # divided by number of opponents (n-1) to be in [0,1]
        normalized_scores: Dict[str, float] = {}
        for eid in order:
            actual = scores[eid] / (n - 1)
            expected_val = expected_sums[eid] / (n - 1)
            normalized_scores[eid] = (actual, expected_val)
        return normalized_scores

    def update_ratings(self, ratings: Dict[str, float], order: List[str]) -> Dict[str, float]:
        """Apply Elo updates for a single event and return new ratings dictionary.

        `order` is a list of entity IDs from best to worst finish.
        """
        pairwise = self.compute_pairwise_scores(ratings, order)
        new_ratings = dict(ratings)
        for eid, (actual, expected_val) in pairwise.items():
            delta = self.k * (actual - expected_val)
            new_ratings[eid] = new_ratings.get(eid, 1500.0) + delta
        return new_ratings


# -----------------------------
# Storage: Google Sheets (optional) or CSV fallback
# -----------------------------

class Storage:
    """Simple storage abstraction. Use Google Sheets if credentials present, otherwise CSV.

    For Google Sheets, user must provide path to a service account JSON credentials file and
    the spreadsheet name.
    """

    def __init__(self, gs_credentials_json: Optional[str] = None, spreadsheet_name: Optional[str] = None):
        self.gs_credentials_json = gs_credentials_json
        self.spreadsheet_name = spreadsheet_name
        self.client = None
        if gs_credentials_json and spreadsheet_name:
            try:
                import gspread
                from oauth2client.service_account import ServiceAccountCredentials
                scope = [
                    'https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive'
                ]
                creds = ServiceAccountCredentials.from_json_keyfile_name(gs_credentials_json, scope)
                self.client = gspread.authorize(creds)
            except Exception as exc:
                logger.exception('Could not initialize Google Sheets client: %s', exc)
                self.client = None

    def load_ratings_from_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def save_ratings_to_csv(self, df: pd.DataFrame, path: str):
        df.to_csv(path, index=False)

    def load_ratings_from_gsheet(self) -> pd.DataFrame:
        if not self.client:
            raise RuntimeError('Google Sheets client not initialized')
        sh = self.client.open(self.spreadsheet_name)
        sheet = sh.sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data)

    def save_ratings_to_gsheet(self, df: pd.DataFrame):
        if not self.client:
            raise RuntimeError('Google Sheets client not initialized')
        sh = self.client.open(self.spreadsheet_name)
        sheet = sh.sheet1
        # Clear and update
        sheet.clear()
        sheet.update([df.columns.values.tolist()] + df.values.tolist())


# -----------------------------
# High-level manager class
# -----------------------------

class F1EloManager:
    """High-level manager tying together fetcher, elo calculator, and storage.

    Example:
        mgr = F1EloManager(cache_dir='f1_cache')
        mgr.bootstrap_from_year(2023)  # optionally initialize all drivers/constructors
        race_df = mgr.fetch_race_results(2023, 7)
        mgr.update_ratings_for_race(race_df)
        mgr.save_ratings_csv('ratings.csv')
    """

    def __init__(self, cache_dir: str = 'f1_cache', k: float = 20.0,
                 gs_credentials_json: Optional[str] = None, spreadsheet_name: Optional[str] = None):
        self.fetcher = FastF1Fetcher(cache_dir=cache_dir)
        self.elo = EloCalculator(k=k)
        self.storage = Storage(gs_credentials_json=gs_credentials_json, spreadsheet_name=spreadsheet_name)

        # in-memory dicts keyed by driverId / constructorId
        self.drivers: Dict[str, DriverModel] = {}
        self.constructors: Dict[str, ConstructorModel] = {}

    # -----------------
    # Initialization helpers
    # -----------------

    def initialize_ratings_from_ergast(self, year: int, rounds: Optional[List[int]] = None):
        """Bootstrap drivers and constructors from Ergast for a given year (default: entire season).

        Note: This doesn't compute ELO history — it simply ensures we have entries for all participants
        with default ratings (1500).
        """
        # Get season schedule
        ergast = fastf1.ergast.Ergast()
        season = ergast.get_season(year)
        races = season.content[0]
        for r in races:
            rnd = int(r.get('round'))
            if rounds and rnd not in rounds:
                continue
            # fetch results if available
            try:
                df = self.fetcher.get_race_results_df(year, rnd)
            except Exception:
                continue
            for _, row in df.iterrows():
                # driver
                driver = row.get('Driver') or {}
                driver_id = driver.get('driverId') or driver.get('code') or str(driver.get('url'))
                if driver_id and driver_id not in self.drivers:
                    model = DriverModel(
                        driver_id=driver_id,
                        given_name=driver.get('givenName'),
                        family_name=driver.get('familyName'),
                        code=driver.get('code'),
                        nationality=driver.get('nationality'),
                        elo=1500.0
                    )
                    self.drivers[driver_id] = model
                # constructor
                constr = row.get('Constructor') or {}
                constructor_id = constr.get('constructorId') or constr.get('name')
                if constructor_id and constructor_id not in self.constructors:
                    cmodel = ConstructorModel(
                        constructor_id=constructor_id,
                        name=constr.get('name'),
                        nationality=constr.get('nationality'),
                        elo=1500.0
                    )
                    self.constructors[constructor_id] = cmodel
        logger.info('Initialized %d drivers and %d constructors', len(self.drivers), len(self.constructors))

    # -----------------
    # Fetch / Update
    # -----------------

    def fetch_race_results(self, year: int, round_num: int) -> pd.DataFrame:
        """Public wrapper returning a normalized DataFrame of race results."""
        df = self.fetcher.get_race_results_df(year, round_num)
        return df

    def _build_ordered_ids(self, df: pd.DataFrame) -> (List[str], List[str]):
        """Given a results DataFrame, return two ordered lists (drivers, constructors) best->worst.

        Uses the Ergast response structure — Driver.driverId and Constructor.constructorId where present.
        Rows with missing position will be placed at the end.
        """
        df_sorted = df.copy()
        if 'position' in df_sorted.columns:
            df_sorted = df_sorted.sort_values(by=['position'], na_position='last')
        drivers_order = []
        constructors_order = []
        for _, row in df_sorted.iterrows():
            driver = row.get('Driver') or {}
            driver_id = driver.get('driverId') or driver.get('code') or str(driver.get('url'))
            drivers_order.append(driver_id)
            constr = row.get('Constructor') or {}
            constructor_id = constr.get('constructorId') or constr.get('name')
            constructors_order.append(constructor_id)
            # ensure presence in dicts
            if driver_id and driver_id not in self.drivers:
                self.drivers[driver_id] = DriverModel(driver_id=driver_id, given_name=driver.get('givenName'),
                                                      family_name=driver.get('familyName'), code=driver.get('code'))
            if constructor_id and constructor_id not in self.constructors:
                self.constructors[constructor_id] = ConstructorModel(constructor_id=constructor_id,
                                                                     name=constr.get('name'))
        return drivers_order, constructors_order

    def update_ratings_for_race(self, results_df: pd.DataFrame, year: Optional[int] = None, round_num: Optional[int] = None):
        """Compute and apply ELO updates for drivers and constructors for a single race DataFrame."""
        drivers_order, constructors_order = self._build_ordered_ids(results_df)

        # Prepare current ratings dicts
        driver_ratings = {did: d.elo for did, d in self.drivers.items()}
        constructor_ratings = {cid: c.elo for cid, c in self.constructors.items()}

        # Update drivers
        updated_drivers = self.elo.update_ratings(driver_ratings, drivers_order)
        for did, new_rating in updated_drivers.items():
            model = self.drivers.get(did)
            if model:
                model.elo = float(new_rating)
                model.last_updated = datetime.utcnow()
            else:
                # create if missing
                self.drivers[did] = DriverModel(driver_id=did, elo=float(new_rating), last_updated=datetime.utcnow())

        # Update constructors
        updated_constructors = self.elo.update_ratings(constructor_ratings, constructors_order)
        for cid, new_rating in updated_constructors.items():
            model = self.constructors.get(cid)
            if model:
                model.elo = float(new_rating)
                model.last_updated = datetime.utcnow()
            else:
                self.constructors[cid] = ConstructorModel(constructor_id=cid, elo=float(new_rating), last_updated=datetime.utcnow())

        logger.info('Updated ratings after race %s round %s', year, round_num)

    # -----------------
    # Export / Import
    # -----------------

    def ratings_to_dataframes(self) -> (pd.DataFrame, pd.DataFrame):
        ddf = pd.DataFrame([{
            'driver_id': d.driver_id,
            'name': d.full_name,
            'code': d.code,
            'elo': d.elo,
            'last_updated': d.last_updated
        } for d in self.drivers.values()])

        cdf = pd.DataFrame([{
            'constructor_id': c.constructor_id,
            'name': c.name,
            'elo': c.elo,
            'last_updated': c.last_updated
        } for c in self.constructors.values()])
        return ddf, cdf

    def save_ratings_csv(self, driver_path: str = 'drivers_ratings.csv', constructor_path: str = 'constructors_ratings.csv'):
        ddf, cdf = self.ratings_to_dataframes()
        self.storage.save_ratings_to_csv(ddf, driver_path)
        self.storage.save_ratings_to_csv(cdf, constructor_path)
        logger.info('Saved ratings to %s and %s', driver_path, constructor_path)

    def load_ratings_csv(self, driver_path: str = 'drivers_ratings.csv', constructor_path: str = 'constructors_ratings.csv'):
        try:
            ddf = self.storage.load_ratings_from_csv(driver_path)
            cdf = self.storage.load_ratings_from_csv(constructor_path)
        except Exception as exc:
            logger.exception('Could not load CSVs: %s', exc)
            return
        for _, row in ddf.iterrows():
            self.drivers[row['driver_id']] = DriverModel(
                driver_id=row['driver_id'],
                given_name=None,
                family_name=row.get('name'),
                code=row.get('code'),
                elo=float(row['elo']),
                last_updated=pd.to_datetime(row.get('last_updated')) if row.get('last_updated') else None
            )
        for _, row in cdf.iterrows():
            self.constructors[row['constructor_id']] = ConstructorModel(
                constructor_id=row['constructor_id'],
                name=row.get('name'),
                elo=float(row['elo']),
                last_updated=pd.to_datetime(row.get('last_updated')) if row.get('last_updated') else None
            )
        logger.info('Loaded %d drivers and %d constructors from CSV', len(self.drivers), len(self.constructors))

    def save_ratings_gsheet(self):
        ddf, cdf = self.ratings_to_dataframes()
        self.storage.save_ratings_to_gsheet(ddf)
        # optionally save constructors to another sheet or second sheet

    def load_ratings_gsheet(self):
        df = self.storage.load_ratings_from_gsheet()
        # Example expects a 'driver_id' column
        for _, row in df.iterrows():
            self.drivers[row['driver_id']] = DriverModel(
                driver_id=row['driver_id'],
                family_name=row.get('name'),
                elo=float(row['elo']),
                last_updated=pd.to_datetime(row.get('last_updated')) if row.get('last_updated') else None
            )


# -----------------------------
# Example command-line usage
# -----------------------------

if __name__ == '__main__':
    # Minimal demonstration (won't run Google Sheets parts)
    mgr = F1EloManager(cache_dir='f1_cache', k=24.0)
    # Try to initialize available drivers from 2023 (fast; limited by what Ergast returns)
    try:
        mgr.initialize_ratings_from_ergast(2023)
    except Exception as exc:
        logger.warning('Bootstrap from Ergast failed: %s', exc)

    # Example: fetch and update for Spain 2023 (round 7)
    try:
        df = mgr.fetch_race_results(2023, 7)
        mgr.update_ratings_for_race(df, year=2023, round_num=7)
        mgr.save_ratings_csv()
    except Exception as exc:
        logger.exception('Demo run failed: %s', exc)


# -----------------------------
# Suggested pyproject.toml and uv commands
# -----------------------------
# pyproject.toml (suggested)
# [project]
# name = "f1-elo"
# version = "0.1.0"
# description = "F1 Elo rating module"
# requires-python = "^3.10"
#
# [project.dependencies]
# fastf1 = "*"
# pandas = "*"
# pydantic = "*"
# gspread = "*" # optional
# oauth2client = "*" # optional
#
# [tool.uv]
# # (uv will produce uv.lock when you sync)
#
# Commands (using uv):
# - initialize project: uv init
# - add deps: uv add fastf1 pandas pydantic
# - run script: uv run f1_elo_module.py
# - lock: uv lock
#
# The above keeps your project reproducible and free (uv is a free tool).
# -----------------------------
