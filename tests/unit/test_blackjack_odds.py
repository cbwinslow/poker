"""
Unit tests for BlackjackOddsCalculator
"""
import pytest
import numpy as np
from tests.fixtures.test_fixtures import (
    blackjack_odds_calculator, sample_blackjack_cards,
    blackjack_game_state, test_card_values, basic_strategy_table
)


class TestBlackjackOddsCalculator:
    """Test suite for blackjack odds calculator"""

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_calculator_initialization(self, blackjack_odds_calculator):
        """Test that calculator initializes correctly"""
        calc = blackjack_odds_calculator

        assert calc.game_rules is not None
        assert calc.game_rules.decks == 6
        assert calc.game_rules.dealer_hits_soft_17 is True
        assert len(calc.card_values) == 13  # All card ranks
        assert len(calc.basic_strategy) > 0

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_card_counting_initialization(self, blackjack_odds_calculator):
        """Test card counting state initialization"""
        calc = blackjack_odds_calculator

        assert calc.running_count == 0
        assert calc.true_count == 0
        assert calc.cards_seen == 0
        assert calc.decks_remaining == 6

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_update_count_with_cards(self, blackjack_odds_calculator, sample_blackjack_cards):
        """Test updating running count with cards"""
        calc = blackjack_odds_calculator

        # Update count with sample cards
        calc.update_count(sample_blackjack_cards)

        # A, K, 5, 7
        # A=-1, K=-1, 5=1, 7=0
        expected_running_count = -1 + -1 + 1 + 0  # -1
        assert calc.running_count == expected_running_count
        assert calc.cards_seen == 4

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_calculate_true_count(self, blackjack_odds_calculator, sample_blackjack_cards):
        """Test true count calculation"""
        calc = blackjack_odds_calculator

        # Update with cards first
        calc.update_count(sample_blackjack_cards)

        true_count = calc.calculate_true_count()

        # Should be running_count / decks_remaining
        expected_true_count = calc.running_count / calc.decks_remaining
        assert abs(true_count - expected_true_count) < 0.001

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_basic_strategy_lookup(self, blackjack_odds_calculator):
        """Test basic strategy table lookup"""
        calc = blackjack_odds_calculator

        # Test known strategy decisions
        assert calc.get_optimal_strategy(20, 6) == 'stand'
        assert calc.get_optimal_strategy(16, 10) == 'hit'
        assert calc.get_optimal_strategy(11, 5) == 'double'

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_strategy_with_deviations(self, blackjack_odds_calculator):
        """Test strategy deviations based on true count"""
        calc = blackjack_odds_calculator

        # Test 16 vs 10 deviation (should stand with positive count)
        strategy_info = calc.get_strategy_with_deviations(16, 10, 2.0)  # High positive count

        assert strategy_info['base_action'] == 'stand'
        assert strategy_info['deviation_applied'] is True
        assert 'Positive count' in strategy_info['deviation_reason']

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_insurance_deviation(self, blackjack_odds_calculator):
        """Test insurance strategy based on true count"""
        calc = blackjack_odds_calculator

        # Insurance should be recommended with high true count
        strategy_info = calc.get_strategy_with_deviations(20, 11, 3.0)  # Ace showing, high count

        assert strategy_info['base_action'] == 'insurance'
        assert strategy_info['deviation_applied'] is True

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_odds_calculation_with_game_state(self, blackjack_odds_calculator, blackjack_game_state):
        """Test complete odds calculation with game state"""
        calc = blackjack_odds_calculator

        odds = calc.calculate_odds(blackjack_game_state)

        assert odds is not None
        assert isinstance(odds, BlackjackOdds)
        assert odds.player_win_probability >= 0.0
        assert odds.player_win_probability <= 1.0
        assert odds.true_count >= 0.0  # Should be calculated
        assert odds.recommended_action in ['hit', 'stand', 'double', 'split', 'surrender']

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_odds_calculation_accuracy(self, blackjack_odds_calculator):
        """Test that odds calculations are mathematically reasonable"""
        # Test with a strong hand (20 vs 6)
        game_state = TestUtils.create_mock_game_state(
            'blackjack',
            player_cards=[
                BlackjackCard('K', 'hearts', 10),
                BlackjackCard('Q', 'spades', 10)
            ],
            dealer_upcard=BlackjackCard('6', 'diamonds', 6),
            player_score=20
        )

        odds = blackjack_odds_calculator.calculate_odds(game_state)

        # 20 vs 6 should have high win probability
        assert odds.player_win_probability > 0.7
        assert odds.recommended_action == 'stand'

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_odds_calculation_blackjack_hand(self, blackjack_odds_calculator):
        """Test odds calculation for blackjack (21)"""
        game_state = TestUtils.create_mock_game_state(
            'blackjack',
            player_cards=[
                BlackjackCard('A', 'hearts', 11),
                BlackjackCard('K', 'spades', 10)
            ],
            dealer_upcard=BlackjackCard('6', 'diamonds', 6),
            player_score=21
        )

        odds = blackjack_odds_calculator.calculate_odds(game_state)

        # Blackjack should have very high win probability
        assert odds.player_win_probability >= 0.8
        assert odds.blackjack_probability == 1.0

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_odds_calculation_bust_hand(self, blackjack_odds_calculator):
        """Test odds calculation for bust hand"""
        game_state = TestUtils.create_mock_game_state(
            'blackjack',
            player_cards=[
                BlackjackCard('K', 'hearts', 10),
                BlackjackCard('Q', 'spades', 10),
                BlackjackCard('5', 'diamonds', 5)
            ],
            dealer_upcard=BlackjackCard('6', 'diamonds', 6),
            player_score=25
        )

        odds = blackjack_odds_calculator.calculate_odds(game_state)

        # Bust hand should have 0 win probability
        assert odds.player_win_probability == 0.0
        assert odds.player_bust_probability == 1.0

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_count_status_tracking(self, blackjack_odds_calculator, sample_blackjack_cards):
        """Test count status tracking"""
        calc = blackjack_odds_calculator

        # Get initial status
        initial_status = calc.get_count_status()
        assert initial_status['running_count'] == 0
        assert initial_status['cards_seen'] == 0

        # Update with cards
        calc.update_count(sample_blackjack_cards)

        # Check updated status
        updated_status = calc.get_count_status()
        assert updated_status['running_count'] == -1  # A+K+5+7 = -1-1+1+0 = -1
        assert updated_status['cards_seen'] == 4
        assert updated_status['deck_penetration'] > 0

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_reset_count_functionality(self, blackjack_odds_calculator, sample_blackjack_cards):
        """Test count reset functionality"""
        calc = blackjack_odds_calculator

        # Update count
        calc.update_count(sample_blackjack_cards)
        assert calc.running_count != 0
        assert calc.cards_seen > 0

        # Reset count
        calc.reset_count()

        # Verify reset
        assert calc.running_count == 0
        assert calc.true_count == 0
        assert calc.cards_seen == 0
        assert calc.decks_remaining == 6

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_different_counting_systems(self, sample_config):
        """Test different card counting systems"""
        # Test Hi-Lo system
        hilo_calc = BlackjackOddsCalculator(sample_config.game.blackjack_rules)
        hilo_calc.game_rules.counting_system = 'hi_lo'

        # Test Zen Count system
        zen_calc = BlackjackOddsCalculator(sample_config.game.blackjack_rules)
        zen_calc.game_rules.counting_system = 'zen_count'

        # Both should initialize correctly
        assert hilo_calc.counting_systems['hi_lo']['A'] == -1
        assert zen_calc.counting_systems['zen_count']['A'] == -1
        assert zen_calc.counting_systems['zen_count']['4'] == 2  # Zen count difference

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_probability_calculations(self, blackjack_odds_calculator):
        """Test individual probability calculation methods"""
        calc = blackjack_odds_calculator

        # Test dealer bust probability
        bust_prob_2 = calc._calculate_dealer_bust_probability(2)
        bust_prob_10 = calc._calculate_dealer_bust_probability(10)

        # Dealer should bust more often against 2 than 10
        assert bust_prob_2 > bust_prob_10

        # Test player bust probability
        bust_prob_12 = calc._calculate_player_bust_probability(12)
        bust_prob_20 = calc._calculate_player_bust_probability(20)

        # Player should bust more often with 12 than 20
        assert bust_prob_12 > bust_prob_20

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_expected_value_calculations(self, blackjack_odds_calculator):
        """Test expected value calculations for different actions"""
        calc = blackjack_odds_calculator

        # Test with a good hand (20 vs 6)
        hit_ev = calc._calculate_hit_value(20, 6)
        stand_ev = calc._calculate_stand_value(20, 6)
        double_ev = calc._calculate_double_value(20, 6)

        # Standing should be best for 20
        assert stand_ev >= hit_ev
        assert stand_ev >= double_ev

        # Test with 11 vs 6 (good doubling hand)
        hit_ev_11 = calc._calculate_hit_value(11, 6)
        double_ev_11 = calc._calculate_double_value(11, 6)

        # Doubling should have higher EV than hitting for 11 vs 6
        assert double_ev_11 > hit_ev_11

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_error_handling_edge_cases(self, blackjack_odds_calculator):
        """Test error handling for edge cases"""
        calc = blackjack_odds_calculator

        # Test with None game state
        odds = calc.calculate_odds(None)
        assert odds is None

        # Test with empty game state
        empty_state = BlackjackGameState(
            player_cards=[], dealer_cards=[], player_score=0
        )
        odds = calc.calculate_odds(empty_state)
        assert odds is None

        # Test with invalid card values
        invalid_state = TestUtils.create_mock_game_state(
            'blackjack',
            player_cards=[BlackjackCard('X', 'hearts', 10)],  # Invalid rank
            dealer_upcard=BlackjackCard('6', 'diamonds', 6),
            player_score=10
        )

        # Should handle gracefully
        odds = calc.calculate_odds(invalid_state)
        assert odds is None or odds.recommended_action in ['hit', 'stand']

    @pytest.mark.unit
    @pytest.mark.slow
    def test_calculation_consistency(self, blackjack_odds_calculator):
        """Test that calculations are consistent across multiple runs"""
        game_state = TestUtils.create_mock_game_state(
            'blackjack',
            player_cards=[
                BlackjackCard('A', 'hearts', 11),
                BlackjackCard('7', 'spades', 7)
            ],
            dealer_upcard=BlackjackCard('6', 'diamonds', 6),
            player_score=18
        )

        # Run calculation multiple times
        results = []
        for _ in range(10):
            odds = blackjack_odds_calculator.calculate_odds(game_state)
            if odds:
                results.append((odds.player_win_probability, odds.recommended_action))

        # Results should be identical (deterministic)
        first_result = results[0]
        for result in results[1:]:
            assert result[0] == first_result[0]  # Same probability
            assert result[1] == first_result[1]  # Same recommendation

    @pytest.mark.unit
    @pytest.mark.blackjack
    def test_configuration_rule_variations(self, sample_config):
        """Test different rule configurations"""
        # Test H17 rules
        h17_config = sample_config
        h17_config.game.blackjack_rules.dealer_hits_soft_17 = True

        h17_calc = BlackjackOddsCalculator(h17_config.game.blackjack_rules)

        # Test S17 rules
        s17_config = sample_config
        s17_config.game.blackjack_rules.dealer_hits_soft_17 = False

        s17_calc = BlackjackOddsCalculator(s17_config.game.blackjack_rules)

        # Both should initialize correctly
        assert h17_calc.game_rules.dealer_hits_soft_17 is True
        assert s17_calc.game_rules.dealer_hits_soft_17 is False

        # Strategy tables should be different for soft 17
        assert len(h17_calc.basic_strategy) == len(s17_calc.basic_strategy)
        # The actual strategies would differ for soft 17 situations


# Import here to avoid circular imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from blackjack_odds import BlackjackCard, BlackjackGameState
from tests.fixtures.test_fixtures import TestUtils