"""
Unit tests for PokerOddsCalculator
"""
import pytest
import numpy as np
import time
from tests.fixtures.test_fixtures import (
    poker_odds_calculator, sample_poker_cards,
    poker_game_state, monte_carlo_test_data
)


class TestPokerOddsCalculator:
    """Test suite for poker odds calculator"""

    @pytest.mark.unit
    @pytest.mark.poker
    def test_calculator_initialization(self, poker_odds_calculator):
        """Test that calculator initializes correctly"""
        calc = poker_odds_calculator

        assert len(calc.ranks) == 13  # All poker ranks
        assert len(calc.suits) == 4   # All suits
        assert len(calc.deck) == 52   # Full deck
        assert len(calc.hand_rankings) == 10  # All hand types

    @pytest.mark.unit
    @pytest.mark.poker
    def test_deck_creation(self, poker_odds_calculator):
        """Test that deck is created correctly"""
        calc = poker_odds_calculator

        # Check deck size
        assert len(calc.deck) == 52

        # Check all ranks are present
        ranks_in_deck = set(card.rank for card in calc.deck)
        assert ranks_in_deck == set(calc.ranks)

        # Check all suits are present
        suits_in_deck = set(card.suit for card in calc.deck)
        assert suits_in_deck == set(calc.suits)

    @pytest.mark.unit
    @pytest.mark.poker
    def test_hand_evaluation_royal_flush(self, poker_odds_calculator, sample_poker_cards):
        """Test royal flush hand evaluation"""
        calc = poker_odds_calculator

        # Create royal flush: A K Q J 10 of hearts
        royal_flush = [
            Card('A', 'hearts'),
            Card('K', 'hearts'),
            Card('Q', 'hearts'),
            Card('J', 'hearts'),
            Card('10', 'hearts')
        ]

        rank, category = calc._evaluate_hand([], royal_flush)
        assert rank == 'royal_flush'
        assert category == 'monster'

    @pytest.mark.unit
    @pytest.mark.poker
    def test_hand_evaluation_straight_flush(self, poker_odds_calculator):
        """Test straight flush hand evaluation"""
        calc = poker_odds_calculator

        # Create straight flush: 9 8 7 6 5 of diamonds
        straight_flush = [
            Card('9', 'diamonds'),
            Card('8', 'diamonds'),
            Card('7', 'diamonds'),
            Card('6', 'diamonds'),
            Card('5', 'diamonds')
        ]

        rank, category = calc._evaluate_hand([], straight_flush)
        assert rank == 'straight_flush'
        assert category == 'monster'

    @pytest.mark.unit
    @pytest.mark.poker
    def test_hand_evaluation_four_of_a_kind(self, poker_odds_calculator):
        """Test four of a kind hand evaluation"""
        calc = poker_odds_calculator

        # Create four of a kind: four Aces + K
        four_aces = [
            Card('A', 'hearts'),
            Card('A', 'diamonds'),
            Card('A', 'clubs'),
            Card('A', 'spades'),
            Card('K', 'hearts')
        ]

        rank, category = calc._evaluate_hand([], four_aces)
        assert rank == 'four_of_a_kind'
        assert category == 'monster'

    @pytest.mark.unit
    @pytest.mark.poker
    def test_hand_evaluation_full_house(self, poker_odds_calculator):
        """Test full house hand evaluation"""
        calc = poker_odds_calculator

        # Create full house: three Aces + two Kings
        full_house = [
            Card('A', 'hearts'),
            Card('A', 'diamonds'),
            Card('A', 'clubs'),
            Card('K', 'hearts'),
            Card('K', 'spades')
        ]

        rank, category = calc._evaluate_hand([], full_house)
        assert rank == 'full_house'
        assert category == 'strong'

    @pytest.mark.unit
    @pytest.mark.poker
    def test_hand_evaluation_incomplete_hand(self, poker_odds_calculator, sample_poker_cards):
        """Test evaluation of incomplete hands"""
        calc = poker_odds_calculator

        # Test with only 2 cards (incomplete)
        rank, category = calc._evaluate_hand(sample_poker_cards[:2], [])
        assert rank == 'incomplete'
        assert category == 'unknown'

    @pytest.mark.unit
    @pytest.mark.poker
    def test_equity_vs_random_calculation(self, poker_odds_calculator, monte_carlo_test_data):
        """Test equity calculation against random hands"""
        calc = poker_odds_calculator

        equity = calc._calculate_equity_vs_random(
            monte_carlo_test_data['player_cards'],
            monte_carlo_test_data['community_cards']
        )

        # Equity should be reasonable for A K high
        assert 0.5 <= equity <= 0.8  # AK should have decent equity

        # Test consistency across multiple runs
        equity2 = calc._calculate_equity_vs_random(
            monte_carlo_test_data['player_cards'],
            monte_carlo_test_data['community_cards']
        )

        # Results should be similar (within 5% due to randomness)
        assert abs(equity - equity2) < 0.05

    @pytest.mark.unit
    @pytest.mark.poker
    @pytest.mark.slow
    def test_equity_calculation_accuracy(self, poker_odds_calculator):
        """Test equity calculation accuracy with known scenarios"""
        calc = poker_odds_calculator

        # Test with pocket Aces pre-flop (should have high equity)
        aces = [Card('A', 'hearts'), Card('A', 'spades')]
        equity_aces = calc._calculate_equity_vs_random(aces, [])

        # Test with 7-2 offsuit pre-flop (should have low equity)
        bad_hand = [Card('7', 'hearts'), Card('2', 'spades')]
        equity_bad = calc._calculate_equity_vs_random(bad_hand, [])

        # Aces should have significantly higher equity than 7-2
        assert equity_aces > equity_bad + 0.3

    @pytest.mark.unit
    @pytest.mark.poker
    def test_pot_odds_calculation(self, poker_odds_calculator, poker_game_state):
        """Test pot odds calculation"""
        calc = poker_odds_calculator

        pot_odds = calc._calculate_pot_odds(poker_game_state)

        # Pot odds should be between 0 and 1
        assert 0 <= pot_odds <= 1

        # Test calculation: call_size / (pot_size + call_size)
        expected_pot_odds = poker_game_state.current_bet / (poker_game_state.pot_size + poker_game_state.current_bet)
        assert abs(pot_odds - expected_pot_odds) < 0.001

    @pytest.mark.unit
    @pytest.mark.poker
    def test_position_advantage_calculation(self, poker_odds_calculator, poker_game_state):
        """Test position advantage calculation"""
        calc = poker_odds_calculator

        # Test different positions
        positions = ['BTN', 'CO', 'MP', 'UTG', 'BB']
        advantages = []

        for pos in positions:
            poker_game_state.player_position = pos
            advantage = calc._calculate_position_advantage(poker_game_state)
            advantages.append(advantage)

        # Button should have highest advantage
        assert advantages[0] == max(advantages)  # BTN should be highest

        # Big blind should have lowest advantage
        assert advantages[-1] == min(advantages)  # BB should be lowest

    @pytest.mark.unit
    @pytest.mark.poker
    def test_fold_equity_calculation(self, poker_odds_calculator, poker_game_state):
        """Test fold equity calculation"""
        calc = poker_odds_calculator

        fold_equity = calc._calculate_fold_equity(poker_game_state)

        # Fold equity should be non-negative
        assert fold_equity >= 0

        # Test with very large bet (should increase fold equity)
        original_bet = poker_game_state.current_bet
        poker_game_state.current_bet = poker_game_state.player_stack  # All-in bet

        high_bet_fold_equity = calc._calculate_fold_equity(poker_game_state)
        poker_game_state.current_bet = original_bet  # Restore

        # All-in bet should generally have higher fold equity
        assert high_bet_fold_equity >= fold_equity

    @pytest.mark.unit
    @pytest.mark.poker
    def test_expected_value_calculation(self, poker_odds_calculator, poker_game_state):
        """Test expected value calculation"""
        calc = poker_odds_calculator

        # Calculate equity first
        equity = calc._calculate_equity_vs_random(
            poker_game_state.player_cards,
            poker_game_state.community_cards
        )

        ev = calc._calculate_expected_value(equity, 0.3, poker_game_state)  # 30% pot odds

        # EV calculation: equity * pot_size - (1-equity) * bet_size
        expected_ev = (equity * poker_game_state.pot_size) - ((1 - equity) * poker_game_state.current_bet)
        assert abs(ev - expected_ev) < 0.01

    @pytest.mark.unit
    @pytest.mark.poker
    def test_hand_ranking_values(self, poker_odds_calculator):
        """Test hand ranking value calculations"""
        calc = poker_odds_calculator

        # Test that better hands have higher values
        royal_flush_value = calc._evaluate_hand_value(
            [Card('A', 'hearts')],  # Player cards
            [Card('K', 'hearts'), Card('Q', 'hearts'), Card('J', 'hearts'), Card('10', 'hearts')]  # Royal flush
        )

        straight_value = calc._evaluate_hand_value(
            [Card('A', 'hearts')],
            [Card('K', 'hearts'), Card('Q', 'hearts'), Card('J', 'hearts'), Card('9', 'hearts')]  # Straight
        )

        assert royal_flush_value > straight_value

    @pytest.mark.unit
    @pytest.mark.poker
    def test_deck_remaining_calculation(self, poker_odds_calculator, sample_poker_cards):
        """Test remaining deck calculation"""
        calc = poker_odds_calculator

        known_cards = sample_poker_cards[:3]  # 3 known cards
        remaining = calc._get_remaining_deck(known_cards)

        # Should have 49 cards remaining (52 - 3)
        assert len(remaining) == 49

        # Known cards should not be in remaining deck
        known_ranks_suits = {(card.rank, card.suit) for card in known_cards}
        remaining_ranks_suits = {(card.rank, card.suit) for card in remaining}
        assert len(remaining_ranks_suits.intersection(known_ranks_suits)) == 0

    @pytest.mark.unit
    @pytest.mark.poker
    def test_card_dealing_simulation(self, poker_odds_calculator):
        """Test card dealing for Monte Carlo simulation"""
        calc = poker_odds_calculator

        # Test dealing cards from deck
        remaining_deck = calc.deck.copy()
        dealt_cards = calc._deal_random_cards(remaining_deck, 2)

        assert len(dealt_cards) == 2
        assert all(card in remaining_deck for card in dealt_cards)

        # Test edge case: not enough cards
        small_deck = calc.deck[:3]  # Only 3 cards
        dealt_cards = calc._deal_random_cards(small_deck, 5)  # Ask for 5

        # Should return empty list when not enough cards
        assert len(dealt_cards) == 0

    @pytest.mark.unit
    @pytest.mark.poker
    def test_effective_hand_strength_calculation(self, poker_odds_calculator, poker_game_state):
        """Test Effective Hand Strength calculation"""
        calc = poker_odds_calculator

        ehs = calc.calculate_effective_hand_strength(
            poker_game_state.player_cards,
            poker_game_state.community_cards
        )

        # EHS should be between 0 and 1
        assert 0 <= ehs <= 1

        # EHS should be similar to raw equity but account for future potential
        raw_equity = calc._calculate_equity_vs_random(
            poker_game_state.player_cards,
            poker_game_state.community_cards
        )

        # EHS should generally be within reasonable range of raw equity
        assert abs(ehs - raw_equity) < 0.3

    @pytest.mark.unit
    @pytest.mark.poker
    def test_opponent_range_analysis(self, poker_odds_calculator, poker_game_state):
        """Test opponent range analysis"""
        calc = poker_odds_calculator

        range_analysis = calc._analyze_opponent_range(poker_game_state)

        # Should return probability distribution
        assert 'tight_range' in range_analysis
        assert 'medium_range' in range_analysis
        assert 'loose_range' in range_analysis

        # Probabilities should sum to approximately 1
        total_prob = sum(range_analysis.values())
        assert 0.9 <= total_prob <= 1.1

    @pytest.mark.unit
    @pytest.mark.poker
    def test_recommendation_logic(self, poker_odds_calculator, poker_game_state):
        """Test action recommendation logic"""
        calc = poker_odds_calculator

        # Test with high equity (should recommend raise)
        high_equity_state = poker_game_state
        # Create a very strong hand
        high_equity_state.player_cards = [Card('A', 'hearts'), Card('A', 'spades')]

        odds = calc.calculate_odds(high_equity_state)
        if odds and odds.hand_category in ['monster', 'strong']:
            assert odds.recommended_action == 'raise'

    @pytest.mark.unit
    @pytest.mark.poker
    def test_cache_functionality(self, poker_odds_calculator):
        """Test equity caching for performance"""
        calc = poker_odds_calculator

        # First calculation
        cards = [Card('A', 'hearts'), Card('K', 'spades')]
        equity1 = calc._calculate_equity_vs_random(cards, [])

        # Check that cache was populated
        assert hasattr(calc, '_equity_cache')

        # Cache should contain the result
        cache_key = f"random_2_0"  # 2 cards, 0 community
        assert cache_key in calc._equity_cache

    @pytest.mark.unit
    @pytest.mark.poker
    def test_error_handling_edge_cases(self, poker_odds_calculator):
        """Test error handling for edge cases"""
        calc = poker_odds_calculator

        # Test with None inputs
        odds = calc.calculate_odds(None)
        assert odds is None

        # Test with empty game state
        empty_state = PokerGameState(
            player_cards=[], community_cards=[], pot_size=0
        )
        odds = calc.calculate_odds(empty_state)
        assert odds is None

        # Test with invalid cards
        invalid_state = PokerGameState(
            player_cards=[Card('X', 'hearts')],  # Invalid rank
            community_cards=[],
            pot_size=100
        )

        # Should handle gracefully
        odds = calc.calculate_odds(invalid_state)
        assert odds is None or odds.recommended_action in ['fold', 'call', 'raise']

    @pytest.mark.unit
    @pytest.mark.poker
    def test_calculation_consistency(self, poker_odds_calculator):
        """Test that calculations are deterministic where expected"""
        calc = poker_odds_calculator

        test_cards = [Card('Q', 'hearts'), Card('J', 'spades')]

        # Run multiple calculations
        results = []
        for _ in range(5):
            odds = calc.calculate_odds(PokerGameState(
                player_cards=test_cards,
                community_cards=[],
                pot_size=100,
                current_bet=20,
                total_bet=20,
                player_stack=500,
                player_position='BTN',
                player_count=6,
                opponent_stacks=[400, 450, 300, 350, 600],
                game_phase='preflop'
            ))
            if odds:
                results.append((odds.hand_rank, odds.recommended_action))

        # Results should be identical (deterministic)
        first_result = results[0]
        for result in results[1:]:
            assert result[0] == first_result[0]  # Same hand rank
            assert result[1] == first_result[1]  # Same recommendation

    @pytest.mark.unit
    @pytest.mark.poker
    def test_specific_hand_odds_calculation(self, poker_odds_calculator, sample_poker_cards):
        """Test calculation of odds for specific hand types"""
        calc = poker_odds_calculator

        specific_odds = calc._calculate_specific_hand_odds(
            sample_poker_cards[:2],  # A, K
            sample_poker_cards[2:5]  # Q, J, 10
        )

        # Should return dictionary with hand type probabilities
        expected_keys = ['royal_flush', 'straight_flush', 'four_of_a_kind',
                        'full_house', 'flush', 'straight', 'three_of_a_kind',
                        'two_pair', 'pair']

        for key in expected_keys:
            assert key in specific_odds
            assert 0 <= specific_odds[key] <= 1  # Probabilities

    @pytest.mark.unit
    @pytest.mark.poker
    def test_bluff_factor_calculation(self, poker_odds_calculator, poker_game_state):
        """Test bluff factor calculation"""
        calc = poker_odds_calculator

        bluff_factor = calc._calculate_bluff_factor(poker_game_state)

        # Bluff factor should be between 0 and 0.5
        assert 0 <= bluff_factor <= 0.5

        # Test with late position (should increase bluff factor)
        original_position = poker_game_state.player_position
        poker_game_state.player_position = 'BTN'

        late_pos_bluff = calc._calculate_bluff_factor(poker_game_state)
        poker_game_state.player_position = original_position

        # Late position should generally have higher bluff factor
        assert late_pos_bluff >= bluff_factor - 0.1  # Allow some tolerance

    @pytest.mark.unit
    @pytest.mark.poker
    def test_hand_strength_categorization(self, poker_odds_calculator):
        """Test hand strength categorization"""
        calc = poker_odds_calculator

        # Test monster hand
        monster_hand = [Card('A', 'hearts'), Card('A', 'spades')]
        monster_state = PokerGameState(player_cards=monster_hand, community_cards=[],
                                     pot_size=100, current_bet=20, total_bet=20,
                                     player_stack=500, player_position='BTN', player_count=6,
                                     opponent_stacks=[400]*5, game_phase='preflop')

        odds = calc.calculate_odds(monster_state)
        assert odds.hand_category in ['monster', 'strong']

        # Test weak hand
        weak_hand = [Card('7', 'hearts'), Card('2', 'spades')]
        weak_state = PokerGameState(player_cards=weak_hand, community_cards=[],
                                  pot_size=100, current_bet=20, total_bet=20,
                                  player_stack=500, player_position='BTN', player_count=6,
                                  opponent_stacks=[400]*5, game_phase='preflop')

        odds = calc.calculate_odds(weak_state)
        assert odds.hand_category in ['weak', 'trash']

    @pytest.mark.unit
    @pytest.mark.poker
    @pytest.mark.slow
    def test_monte_carlo_convergence(self, poker_odds_calculator):
        """Test that Monte Carlo simulations converge to reasonable values"""
        calc = poker_odds_calculator

        # Test with different simulation counts
        cards = [Card('A', 'hearts'), Card('K', 'spades')]

        # Small simulation count
        calc.simulation_count = 100
        equity_small = calc._calculate_equity_vs_random(cards, [])

        # Large simulation count
        calc.simulation_count = 10000
        equity_large = calc._calculate_equity_vs_random(cards, [])

        # Results should be reasonably close (within 10%)
        assert abs(equity_small - equity_large) < 0.1

        # Both should be in reasonable range for AK
        for equity in [equity_small, equity_large]:
            assert 0.55 <= equity <= 0.75


# Import here to avoid circular imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from poker_odds import Card, PokerGameState