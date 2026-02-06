"""Scraper modules for the Universal Data Scraper."""

from .base import ScraperModule, ScraperRegistry
from .images import ImageScraper
from .chess_positions import ChessPositionScraper
from .numeric_relations import NumericRelationScraper
from .experiments import ExperimentScraper
from .social import SocialScraper

__all__ = [
    "ScraperModule",
    "ScraperRegistry",
    "ImageScraper",
    "ChessPositionScraper",
    "NumericRelationScraper",
    "ExperimentScraper",
    "SocialScraper",
]
