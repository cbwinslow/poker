"""
Configuration settings for the Poker/Blackjack AI Agent
"""
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Any


@dataclass
class ScreenConfig:
    """Screen capture configuration"""
    monitor: int = 1
    capture_rate: int = 10  # FPS
    region: Tuple[int, int, int, int] = None  # (x, y, width, height)
    adaptive_delay: bool = True  # Enable adaptive delay based on game state
    idle_delay: float = 0.1  # Delay when no game detected (seconds)
    active_delay: float = 0.05  # Delay when game is active (seconds)


@dataclass
class BlackjackRules:
    """Blackjack game rules configuration"""
    decks: int = 6  # Number of decks
    dealer_hits_soft_17: bool = True  # H17 vs S17
    double_after_split: bool = True  # DAS allowed
    surrender_allowed: bool = True  # Late surrender available
    double_down_rules: str = 'any'  # 'any', '9-11', '10-11'
    blackjack_payout: float = 1.5  # 3:2 payout
    max_splits: int = 3  # Maximum number of splits allowed
    penetration_warning: float = 0.75  # Warn when penetration exceeds this

@dataclass
class PokerRules:
    """Poker game rules configuration"""
    game_type: str = 'texas_holdem'  # Game variant
    stakes: str = 'NLHE'  # NLHE, PLO, etc.
    blinds_structure: Dict[str, float] = None  # Small blind, big blind
    max_buyin: float = 100.0  # Maximum buy-in
    rake_structure: Dict[str, float] = None  # Rake percentage/cap
    time_bank: int = 30  # Time bank in seconds

@dataclass
class GameConfig:
    """Game detection configuration"""
    poker_regions: Dict[str, Tuple[int, int, int, int]] = None
    blackjack_regions: Dict[str, Tuple[int, int, int, int]] = None
    card_detection_threshold: float = 0.8
    ocr_confidence: float = 0.7
    auto_pause_on_idle: bool = True  # Auto-pause detection when no game detected
    idle_timeout_seconds: int = 30  # Seconds of idle before auto-pause
    detection_sensitivity: str = 'normal'  # 'low', 'normal', 'high'

    # Enhanced game rules
    blackjack_rules: BlackjackRules = None
    poker_rules: PokerRules = None

    # Card counting configuration
    counting_system: str = 'hi_lo'  # 'hi_lo', 'zen_count', 'wong_halves'
    count_decks_separately: bool = False  # For multi-deck games
    true_count_precision: float = 0.1  # Precision for true count calculation

    # Advanced features
    enable_composition_strategy: bool = True  # Use composition-dependent strategy
    enable_opponent_modeling: bool = True  # Enable poker opponent modeling
    simulation_count: int = 10000  # Number of Monte Carlo simulations
    adaptive_simulation: bool = True  # Adjust simulation count based on available time


@dataclass
class UIConfig:
    """UI overlay configuration"""
    overlay_alpha: float = 0.9
    font_size: int = 12
    odds_display_position: Tuple[int, int] = (10, 10)
    show_probabilities: bool = True
    show_recommendations: bool = True
    show_hand_rank: bool = True
    show_confidence: bool = False
    enable_hotkeys: bool = True
    toggle_hotkey: str = 'ctrl+alt+t'  # Hotkey to toggle agent on/off
    pause_hotkey: str = 'ctrl+alt+p'   # Hotkey to pause/resume detection
    minimize_hotkey: str = 'ctrl+alt+m'  # Hotkey to minimize/restore overlay
    enable_minimize: bool = True  # Allow overlay minimization
    auto_minimize_on_idle: bool = False  # Auto-minimize when idle


@dataclass
class LoggingConfig:
    """Logging configuration"""
    log_level: str = 'INFO'  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    log_to_file: bool = True
    log_filename: str = 'poker_ai_agent.log'
    max_log_size_mb: int = 10


@dataclass
class Config:
    """Main configuration class"""
    screen: ScreenConfig
    game: GameConfig
    ui: UIConfig
    logging: LoggingConfig

    def __init__(self):
        self.screen = ScreenConfig()
        self.game = GameConfig()
        self.ui = UIConfig()
        self.logging = LoggingConfig()
        self._load_defaults()

    def _load_defaults(self):
        """Load default configurations"""
        # Set up default poker regions (these would need calibration for specific games)
        self.game.poker_regions = {
            'player_cards': (100, 400, 200, 100),
            'community_cards': (300, 300, 400, 100),
            'pot_info': (500, 100, 200, 50),
            'player_info': (50, 500, 150, 100)
        }

        # Set up default blackjack regions
        self.game.blackjack_regions = {
            'player_cards': (200, 400, 300, 100),
            'dealer_cards': (200, 200, 300, 100),
            'dealer_upcard': (200, 200, 150, 100),
            'score_areas': (500, 200, 100, 200)
        }

        # Initialize blackjack rules with common casino defaults
        if self.game.blackjack_rules is None:
            self.game.blackjack_rules = BlackjackRules(
                decks=6,
                dealer_hits_soft_17=True,  # H17 (common)
                double_after_split=True,
                surrender_allowed=True,
                double_down_rules='any',
                blackjack_payout=1.5,
                max_splits=3,
                penetration_warning=0.75
            )

        # Initialize poker rules with standard defaults
        if self.game.poker_rules is None:
            self.game.poker_rules = PokerRules(
                game_type='texas_holdem',
                stakes='NLHE',
                blinds_structure={'small_blind': 1.0, 'big_blind': 2.0},
                max_buyin=100.0,
                rake_structure={'percentage': 0.05, 'cap': 3.0},
                time_bank=30
            )


    def _load_defaults(self):
        """Load default configurations"""
        # Set up default poker regions (these would need calibration for specific games)
        self.game.poker_regions = {
            'player_cards': (100, 400, 200, 100),
            'community_cards': (300, 300, 400, 100),
            'pot_info': (500, 100, 200, 50),
            'player_info': (50, 500, 150, 100)
        }

        # Set up default blackjack regions
        self.game.blackjack_regions = {
            'player_cards': (200, 400, 300, 100),
            'dealer_cards': (200, 200, 300, 100),
            'dealer_upcard': (200, 200, 150, 100),
            'score_areas': (500, 200, 100, 200)
        }


# Global configuration instance
config = Config()