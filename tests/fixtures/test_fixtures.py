"""
Test fixtures and utilities for the AI Blackjack Poker Assistant test suite
"""
import pytest
import numpy as np
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from config.config import Config, BlackjackRules, PokerRules
from blackjack_odds import BlackjackCard, BlackjackGameState, BlackjackOdds, BlackjackOddsCalculator
from poker_odds import Card, PokerGameState, PokerOdds, PokerOddsCalculator
from historical_tracker import DatabaseManager, GameSession, HandResult


@pytest.fixture
def sample_config():
    """Create a sample configuration for testing"""
    config = Config()

    # Set up test blackjack rules
    config.game.blackjack_rules = BlackjackRules(
        decks=6,
        dealer_hits_soft_17=True,
        double_after_split=True,
        surrender_allowed=True,
        double_down_rules='any',
        blackjack_payout=1.5,
        max_splits=3,
        penetration_warning=0.75
    )

    # Set up test poker rules
    config.game.poker_rules = PokerRules(
        game_type='texas_holdem',
        stakes='NLHE',
        blinds_structure={'small_blind': 1.0, 'big_blind': 2.0},
        max_buyin=100.0,
        rake_structure={'percentage': 0.05, 'cap': 3.0},
        time_bank=30
    )

    return config


@pytest.fixture
def blackjack_odds_calculator(sample_config):
    """Create a blackjack odds calculator for testing"""
    return BlackjackOddsCalculator(sample_config.game.blackjack_rules)


@pytest.fixture
def poker_odds_calculator():
    """Create a poker odds calculator for testing"""
    return PokerOddsCalculator()


@pytest.fixture
def database_manager():
    """Create a test database manager"""
    # Use in-memory database for tests
    return DatabaseManager(':memory:')


@pytest.fixture
def sample_blackjack_cards():
    """Create sample blackjack cards for testing"""
    cards = [
        BlackjackCard('A', 'hearts', 11),
        BlackjackCard('K', 'spades', 10),
        BlackjackCard('5', 'diamonds', 5),
        BlackjackCard('7', 'clubs', 7)
    ]
    return cards


@pytest.fixture
def sample_poker_cards():
    """Create sample poker cards for testing"""
    cards = [
        Card('A', 'hearts'),
        Card('K', 'spades'),
        Card('Q', 'diamonds'),
        Card('J', 'clubs'),
        Card('10', 'hearts'),
        Card('5', 'spades'),
        Card('3', 'diamonds')
    ]
    return cards


@pytest.fixture
def blackjack_game_state(sample_blackjack_cards):
    """Create a sample blackjack game state"""
    player_cards = sample_blackjack_cards[:2]  # A, K
    dealer_cards = sample_blackjack_cards[2:]  # 5, 7

    return BlackjackGameState(
        player_cards=player_cards,
        dealer_cards=dealer_cards,
        player_score=21,
        dealer_upcard=dealer_cards[0] if dealer_cards else None,
        deck_count=6,
        current_bet=10.0,
        player_bankroll=1000.0,
        player_can_double=True,
        player_can_split=False,
        true_count=0.0
    )


@pytest.fixture
def poker_game_state(sample_poker_cards):
    """Create a sample poker game state"""
    player_cards = sample_poker_cards[:2]  # A, K
    community_cards = sample_poker_cards[2:5]  # Q, J, 10

    return PokerGameState(
        player_cards=player_cards,
        community_cards=community_cards,
        pot_size=100.0,
        current_bet=20.0,
        total_bet=20.0,
        player_stack=500.0,
        player_position='BTN',
        player_count=6,
        opponent_stacks=[400, 450, 300, 350, 600],
        game_phase='flop'
    )


@pytest.fixture
def test_session(database_manager):
    """Create a test gaming session"""
    session = GameSession(
        session_id='test_session_123',
        game_type='blackjack',
        start_time=1234567890.0,
        hands_played=0,
        total_profit_loss=0.0
    )
    database_manager.insert_session(session)
    return session


@pytest.fixture
def sample_hand_result():
    """Create a sample hand result for testing"""
    return HandResult(
        game_type='blackjack',
        timestamp=1234567890.0,
        predicted_action='stand',
        actual_outcome='stand',
        expected_value=0.5,
        actual_profit_loss=10.0,
        game_state_summary={'player_score': 20, 'dealer_upcard': 6},
        confidence_score=0.85
    )


@pytest.fixture
def test_card_values():
    """Standard card values for testing"""
    return {
        '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
        '7': 0, '8': 0, '9': 0, '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
    }


@pytest.fixture
def basic_strategy_table():
    """Sample basic strategy table for testing"""
    return {
        (20, 6): 'stand',
        (16, 10): 'hit',
        (11, 5): 'double',
        (13, 2): 'stand',
        (12, 3): 'hit'
    }


@pytest.fixture
def monte_carlo_test_data():
    """Test data for Monte Carlo validation"""
    return {
        'player_cards': [Card('A', 'hearts'), Card('K', 'spades')],
        'community_cards': [],
        'expected_equity_range': (0.6, 0.8),  # Expected equity should be in this range
        'simulation_count': 1000
    }


@pytest.fixture
def performance_test_config():
    """Configuration for performance testing"""
    return {
        'target_latency_ms': 200,
        'max_memory_mb': 100,
        'min_accuracy_threshold': 0.8,
        'stress_test_duration': 30  # seconds
    }


@pytest.fixture
def validation_scenarios():
    """Pre-defined scenarios for validation testing"""
    return [
        {
            'name': 'blackjack_perfect_pair',
            'game_type': 'blackjack',
            'player_cards': [BlackjackCard('K', 'hearts', 10), BlackjackCard('K', 'spades', 10)],
            'dealer_upcard': BlackjackCard('6', 'diamonds', 6),
            'expected_action': 'split',
            'confidence_threshold': 0.9
        },
        {
            'name': 'poker_premium_hand',
            'game_type': 'poker',
            'player_cards': [Card('A', 'hearts'), Card('K', 'spades')],
            'community_cards': [],
            'expected_equity_min': 0.65,
            'confidence_threshold': 0.85
        },
        {
            'name': 'blackjack_soft_17',
            'game_type': 'blackjack',
            'player_cards': [BlackjackCard('A', 'hearts', 11), BlackjackCard('6', 'spades', 6)],
            'dealer_upcard': BlackjackCard('7', 'diamonds', 7),
            'expected_action': 'hit',
            'confidence_threshold': 0.8
        }
    ]


# Test utilities
class TestUtils:
    """Utility functions for testing"""

    @staticmethod
    def assert_equity_in_range(equity: float, expected_range: tuple, tolerance: float = 0.05):
        """Assert that equity is within expected range"""
        min_val, max_val = expected_range
        assert min_val - tolerance <= equity <= max_val + tolerance, \
            f"Equity {equity:.3f} not in expected range {expected_range}"

    @staticmethod
    def assert_action_matches_strategy(action: str, expected_action: str, game_state: Any):
        """Assert that action matches expected strategy"""
        assert action == expected_action, \
            f"Action '{action}' does not match expected '{expected_action}' for {game_state}"

    @staticmethod
    def create_mock_game_state(game_type: str, **kwargs) -> Any:
        """Create a mock game state for testing"""
        if game_type == 'blackjack':
            return BlackjackGameState(
                player_cards=kwargs.get('player_cards', []),
                dealer_cards=kwargs.get('dealer_cards', []),
                player_score=kwargs.get('player_score', 0),
                dealer_upcard=kwargs.get('dealer_upcard'),
                deck_count=kwargs.get('deck_count', 6),
                current_bet=kwargs.get('current_bet', 10.0),
                player_bankroll=kwargs.get('player_bankroll', 1000.0),
                player_can_double=kwargs.get('player_can_double', True),
                player_can_split=kwargs.get('player_can_split', False),
                true_count=kwargs.get('true_count', 0.0)
            )
        elif game_type == 'poker':
            return PokerGameState(
                player_cards=kwargs.get('player_cards', []),
                community_cards=kwargs.get('community_cards', []),
                pot_size=kwargs.get('pot_size', 100.0),
                current_bet=kwargs.get('current_bet', 20.0),
                total_bet=kwargs.get('total_bet', 20.0),
                player_stack=kwargs.get('player_stack', 500.0),
                player_position=kwargs.get('player_position', 'BTN'),
                player_count=kwargs.get('player_count', 6),
                opponent_stacks=kwargs.get('opponent_stacks', []),
                game_phase=kwargs.get('game_phase', 'preflop')
            )

    @staticmethod
    def calculate_expected_accuracy(predictions: List[str], actuals: List[str]) -> float:
        """Calculate accuracy from prediction/actual pairs"""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")

        correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
        return correct / len(predictions) if predictions else 0.0

    @staticmethod
    def assert_performance_requirements(performance_data: Dict[str, Any], requirements: Dict[str, float]):
        """Assert that performance meets requirements"""
        for metric, threshold in requirements.items():
            actual_value = performance_data.get(metric, 0)
            assert actual_value >= threshold, \
                f"Performance {metric}: {actual_value:.3f} < required {threshold:.3f}"


# Pytest configuration fixtures
@pytest.fixture(scope="session")
def test_db():
    """Create a test database for the entire session"""
    db = DatabaseManager(':memory:')
    yield db
    # Cleanup would go here if needed


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically cleanup after each test"""
    yield
    # Clean up any resources if needed


# Mock objects for testing
class MockScreenCapture:
    """Mock screen capture for testing"""

    def __init__(self):
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.monitoring = False

    def start_monitoring(self):
        self.monitoring = True
        return True

    def stop_monitoring(self):
        self.monitoring = False

    def get_current_frame(self):
        return self.frame.copy() if self.monitoring else None


@pytest.fixture
def mock_screen_capture():
    """Mock screen capture for testing"""
    return MockScreenCapture()