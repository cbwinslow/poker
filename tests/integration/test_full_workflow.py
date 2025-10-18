
"""
Integration tests for complete AI assistant workflows
"""
import pytest
import time
import os
import sys
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tests.fixtures.test_fixtures import (
    sample_config, blackjack_odds_calculator, poker_odds_calculator,
    database_manager, test_session, sample_hand_result, TestUtils
)
from blackjack_odds import BlackjackCard, BlackjackGameState
from poker_odds import Card, PokerGameState
from historical_tracker import HandResult, DatabaseManager


class TestFullWorkflowIntegration:
    """Integration tests for complete workflows"""

    @pytest.mark.integration
    def test_blackjack_complete_workflow(self, sample_config, database_manager):
        """Test complete blackjack workflow from detection to recommendation"""
        # Initialize components
        calc = BlackjackOddsCalculator(sample_config.game.blackjack_rules)
        db = database_manager

        # Create test game state
        game_state = BlackjackGameState(
            player_cards=[
                BlackjackCard('A', 'hearts', 11),
                BlackjackCard('K', 'spades', 10)
            ],
            dealer_cards=[BlackjackCard('6', 'diamonds', 6)],
            player_score=21,
            dealer_upcard=BlackjackCard('6', 'diamonds', 6),
            current_bet=25.0,
            player_bankroll=1000.0
        )

        # Step 1: Calculate odds
        odds = calc.calculate_odds(game_state)
        assert odds is not None

        # Step 2: Validate recommendation
        assert odds.recommended_action == 'stand'
        assert odds.player_win_probability > 0.8

        # Step 3: Record in database
        hand_result = HandResult(
            game_type='blackjack',
            timestamp=time.time(),
            predicted_action=odds.recommended_action,
            actual_outcome='stand',  # Simulate correct outcome
            expected_value=odds.player_win_probability,
            actual_profit_loss=25.0,  # Blackjack pays 1.5x
            game_state_summary={
                'player_score': game_state.player_score,
                'dealer_upcard': game_state.dealer_upcard.value if game_state.dealer_upcard else 0
            },
            confidence_score=odds.player_win_probability
        )

        hand_id = db.insert_hand_result(hand_result)
        assert hand_id > 0

        # Step 4: Verify database storage
        performance = db.get_model_performance('blackjack')
        assert performance['total_hands'] >= 1
        assert performance['accuracy'] >= 0.8

        print("✅ Blackjack workflow integration test passed")

    @pytest.mark.integration
    def test_poker_complete_workflow(self, poker_odds_calculator, database_manager):
        """Test complete poker workflow from detection to recommendation"""
        # Initialize components
        calc = PokerOddsCalculator()
        db = database_manager

        # Create test game state
        game_state = PokerGameState(
            player_cards=[Card('A', 'hearts'), Card('K', 'spades')],
            community_cards=[Card('Q', 'hearts'), Card('J', 'hearts'), Card('10', 'hearts')],
            pot_size=100,
            current_bet=20,
            total_bet=20,
            player_stack=500,
            player_position='BTN',
            player_count=6,
            opponent_stacks=[400]*5,
            game_phase='flop'
        )

        # Step 1: Calculate odds
        odds = calc.calculate_odds(game_state)
        assert odds is not None

        # Step 2: Validate recommendation
        assert odds.recommended_action in ['fold', 'call', 'raise']
        assert odds.equity_vs_random > 0.5  # AK with royal flush draw should be strong

        # Step 3: Record in database
        hand_result = HandResult(
            game_type='poker',
            timestamp=time.time(),
            predicted_action=odds.recommended_action,
            actual_outcome=odds.recommended_action,  # Simulate correct outcome
            expected_value=odds.equity_vs_random,
            actual_profit_loss=15.0,
            game_state_summary={
                'hand_category': odds.hand_category,
                'pot_odds': odds.pot_odds
            },
            confidence_score=odds.hand_strength
        )

        hand_id = db.insert_hand_result(hand_result)
        assert hand_id > 0

        # Step 4: Verify database storage
        performance = db.get_model_performance('poker')
        assert performance['total_hands'] >= 1

        print("✅ Poker workflow integration test passed")

    @pytest.mark.integration
    def test_cross_game_workflow(self, sample_config, database_manager):
        """Test workflow across both poker and blackjack"""
        # Initialize components
        blackjack_calc = BlackjackOddsCalculator(sample_config.game.blackjack_rules)
        poker_calc = PokerOddsCalculator()
        db = database_manager

        results = []

        # Test multiple hands of each game
        for i in range(5):
            # Blackjack hand
            blackjack_state = BlackjackGameState(
                player_cards=[
                    BlackjackCard('10', 'hearts', 10),
                    BlackjackCard('7', 'spades', 7)
                ],
                dealer_cards=[BlackjackCard('8', 'diamonds', 8)],
                player_score=17,
                dealer_upcard=BlackjackCard('8', 'diamonds', 8)
            )

            blackjack_odds = blackjack_calc.calculate_odds(blackjack_state)
            if blackjack_odds:
                blackjack_result = HandResult(
                    game_type='blackjack',
                    timestamp=time.time(),
                    predicted_action=blackjack_odds.recommended_action,
                    actual_outcome=blackjack_odds.recommended_action,
                    expected_value=blackjack_odds.player_win_probability,
                    actual_profit_loss=10.0,
                    game_state_summary={'player_score': 17},
                    confidence_score=0.8
                )
                db.insert_hand_result(blackjack_result)
                results.append('blackjack')

            # Poker hand
            poker_state = PokerGameState(
                player_cards=[Card('Q', 'hearts'), Card('J', 'spades')],
                community_cards=[Card('K', 'hearts'), Card('10', 'hearts')],
                pot_size=100,
                current_bet=20,
                total_bet=20,
                player_stack=500,
                player_position='BTN',
                player_count=6,
                opponent_stacks=[400]*5,
                game_phase='flop'
            )

            poker_odds = poker_calc.calculate_odds(poker_state)
            if poker_odds:
                poker_result = HandResult(
                    game_type='poker',
                    timestamp=time.time(),
                    predicted_action=poker_odds.recommended_action,
                    actual_outcome=poker_odds.recommended_action,
                    expected_value=poker_odds.equity_vs_random,
                    actual_profit_loss=10.0,
                    game_state_summary={'hand_category': poker_odds.hand_category},
                    confidence_score=0.8
                )
                db.insert_hand_result(poker_result)
                results.append('poker')

        # Verify both games were processed
        assert 'blackjack' in results
        assert 'poker' in results

        # Verify database has records from both games
        blackjack_perf = db.get_model_performance('blackjack')
        poker_perf = db.get_model_performance('poker')

        assert blackjack_perf['total_hands'] >= 3  # At least some blackjack hands
        assert poker_perf['total_hands'] >= 3     # At least some poker hands

        print("✅ Cross-game workflow integration test passed")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_integration(self, sample_config):
        """Test performance integration across components"""
        # Initialize all components
        blackjack_calc = BlackjackOddsCalculator(sample_config.game.blackjack_rules)
        poker_calc = PokerOddsCalculator()
        db = DatabaseManager(':memory:')

        start_time = time.time()

        # Simulate realistic usage pattern
        for i in range(20):
            if i % 2 == 0:
                # Blackjack calculation
                state = BlackjackGameState(
                    player_cards=[
                        BlackjackCard('A', 'hearts', 11),
                        BlackjackCard(str((i % 10) + 2), 'spades', (i % 10) + 2)
                    ],
                    dealer_cards=[BlackjackCard('6', 'diamonds', 6)],
                    player_score=21,
                    dealer_upcard=BlackjackCard('6', 'diamonds', 6)
                )
                odds = blackjack_calc.calculate_odds(state)

                if odds:
                    hand_result = HandResult(
                        game_type='blackjack',
                        timestamp=time.time(),
                        predicted_action=odds.recommended_action,
                        actual_outcome=odds.recommended_action,
                        expected_value=odds.player_win_probability,
                        actual_profit_loss=10.0,
                        game_state_summary={'player_score': state.player_score},
                        confidence_score=0.8
                    )
                    db.insert_hand_result(hand_result)
            else:
                # Poker calculation
                state = PokerGameState(
                    player_cards=[
                        Card(['A', 'K', 'Q', 'J'][i % 4], 'hearts'),
                        Card(['K', 'Q', 'J', '10'][i % 4], 'spades')
                    ],
                    community_cards=[],
                    pot_size=100,
                    current_bet=20,
                    total_bet=20,
                    player_stack=500,
                    player_position='BTN',
                    player_count=6,
                    opponent_stacks=[400]*5,
                    game_phase='preflop'
                )
                odds = poker_calc.calculate_odds(state)

                if odds:
                    hand_result = HandResult(
                        game_type='poker',
                        timestamp=time.time(),
                        predicted_action=odds.recommended_action,
                        actual_outcome=odds.recommended_action,
                        expected_value=odds.equity_vs_random,
                        actual_profit_loss=10.0,
                        game_state_summary={'hand_category': odds.hand_category},
                        confidence_score=0.8
                    )
                    db.insert_hand_result(hand_result)

        total_time = time.time() - start_time

        # Performance assertions
        assert total_time < 10, f"Integration test too slow: {total_time:.2f}s for 20 calculations"

        # Verify database operations
        blackjack_perf = db.get_model_performance('blackjack')
        poker_perf = db.get_model_performance('poker')

        total_hands = blackjack_perf['total_hands'] + poker_perf['total_hands']
        assert total_hands >= 15  # Should have processed most hands

        print(f"✅ Performance integration test passed: {total_time:.2f}s for {total_hands} calculations")

    @pytest.mark.integration
    def test_error_handling_integration(self, sample_config):
        """Test error handling across integrated components"""
        blackjack_calc = BlackjackOddsCalculator(sample_config.game.blackjack_rules)
        poker_calc = PokerOddsCalculator()
        db = DatabaseManager(':memory:')

        # Test with invalid inputs
        invalid_states = [
            None,  # None input
            BlackjackGameState(player_cards=[], dealer_cards=[]),  # Empty state
            PokerGameState(player_cards=[], community_cards=[])     # Empty state
        ]

        error_count = 0

       