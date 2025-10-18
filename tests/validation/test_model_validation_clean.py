"""
Clean validation tests for AI predictions vs actual outcomes
"""
import pytest
import time
import json
from typing import Dict, List, Any
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from tests.fixtures.test_fixtures import (
    sample_config, blackjack_odds_calculator, poker_odds_calculator,
    database_manager, test_session, sample_hand_result, validation_scenarios, TestUtils
)
from blackjack_odds import BlackjackCard, BlackjackGameState, BlackjackOdds
from poker_odds import Card, PokerGameState, PokerOdds
from historical_tracker import HandResult, DatabaseManager


class TestModelValidation:
    """Validation tests that track actual vs expected outcomes"""

    @pytest.mark.validation
    @pytest.mark.blackjack
    def test_blackjack_strategy_validation(self, blackjack_odds_calculator, validation_scenarios):
        """Validate blackjack strategy against known correct plays"""
        calc = blackjack_odds_calculator
        validation_results = []

        for scenario in validation_scenarios:
            if scenario['game_type'] != 'blackjack':
                continue

            # Create game state for scenario
            game_state = TestUtils.create_mock_game_state(
                'blackjack',
                player_cards=scenario['player_cards'],
                dealer_upcard=scenario['dealer_upcard'],
                player_score=sum(card.value for card in scenario['player_cards'])
            )

            # Get AI prediction
            odds = calc.calculate_odds(game_state)

            if odds:
                prediction = odds.recommended_action
                expected = scenario['expected_action']
                confidence = odds.player_win_probability

                result = {
                    'scenario': scenario['name'],
                    'prediction': prediction,
                    'expected': expected,
                    'correct': prediction == expected,
                    'confidence': confidence,
                    'timestamp': time.time()
                }

                validation_results.append(result)

                # Assert correctness if confidence is high enough
                if scenario['confidence_threshold'] > 0:
                    assert prediction == expected, f"Strategy error in {scenario['name']}: predicted {prediction}, expected {expected}"

        # Calculate overall accuracy
        correct_predictions = sum(1 for r in validation_results if r['correct'])
        accuracy = correct_predictions / len(validation_results) if validation_results else 0

        print(f"Blackjack validation accuracy: {accuracy:.2%} ({correct_predictions}/{len(validation_results)})")

        # Overall accuracy should meet minimum threshold
        assert accuracy >= 0.8, f"Blackjack strategy accuracy {accuracy:.2%} below 80% threshold"

    @pytest.mark.validation
    @pytest.mark.poker
    def test_poker_equity_validation(self, poker_odds_calculator, validation_scenarios):
        """Validate poker equity calculations against expected ranges"""
        calc = poker_odds_calculator
        validation_results = []

        for scenario in validation_scenarios:
            if scenario['game_type'] != 'poker':
                continue

            # Create game state for scenario
            game_state = TestUtils.create_mock_game_state(
                'poker',
                player_cards=scenario['player_cards'],
                community_cards=scenario.get('community_cards', [])
            )

            # Get AI prediction
            odds = calc.calculate_odds(game_state)

            if odds:
                equity = odds.equity_vs_random
                expected_min = scenario['expected_equity_min']
                confidence = odds.hand_strength

                result = {
                    'scenario': scenario['name'],
                    'equity': equity,
                    'expected_min': expected_min,
                    'correct': equity >= expected_min,
                    'confidence': confidence,
                    'timestamp': time.time()
                }

                validation_results.append(result)

                # Assert equity meets minimum threshold
                assert equity >= expected_min, f"Equity too low in {scenario['name']}: {equity:.3f} < {expected_min:.3f}"

        # Calculate validation metrics
        correct_predictions = sum(1 for r in validation_results if r['correct'])
        accuracy = correct_predictions / len(validation_results) if validation_results else 0

        print(f"Poker equity validation accuracy: {accuracy:.2%} ({correct_predictions}/{len(validation_results)})")

    @pytest.mark.validation
    @pytest.mark.slow
    def test_monte_carlo_accuracy_validation(self, poker_odds_calculator):
        """Validate Monte Carlo simulation accuracy with statistical tests"""
        calc = poker_odds_calculator

        # Test scenarios with known expected equity ranges
        test_scenarios = [
            {
                'name': 'premium_pair',
                'cards': [Card('A', 'hearts'), Card('A', 'spades')],
                'expected_range': (0.75, 0.85)
            },
            {
                'name': 'weak_ace',
                'cards': [Card('A', 'hearts'), Card('2', 'spades')],
                'expected_range': (0.55, 0.70)
            },
            {
                'name': 'weak_hand',
                'cards': [Card('7', 'hearts'), Card('2', 'spades')],
                'expected_range': (0.30, 0.45)
            }
        ]

        validation_results = []

        for scenario in test_scenarios:
            # Run multiple simulations to validate consistency
            equity_results = []

            for _ in range(10):
                equity = calc._calculate_equity_vs_random(scenario['cards'], [])
                equity_results.append(equity)

            # Calculate statistics
            mean_equity = sum(equity_results) / len(equity_results)
            variance = sum((x - mean_equity) ** 2 for x in equity_results) / len(equity_results)
            std_dev = variance ** 0.5

            expected_min, expected_max = scenario['expected_range']

            result = {
                'scenario': scenario['name'],
                'mean_equity': mean_equity,
                'std_dev': std_dev,
                'min_equity': min(equity_results),
                'max_equity': max(equity_results),
                'expected_range': scenario['expected_range'],
                'within_range': expected_min <= mean_equity <= expected_max,
                'timestamp': time.time()
            }

            validation_results.append(result)

            # Validate that mean equity is within expected range
            assert expected_min <= mean_equity <= expected_max, f"Mean equity {mean_equity:.3f} not in expected range {scenario['expected_range']} for {scenario['name']}"

            # Validate that standard deviation is reasonable (not too much variance)
            assert std_dev < 0.1, f"High variance in {scenario['name']}: std_dev = {std_dev:.3f}"

        print("Monte Carlo accuracy validation completed successfully")

    @pytest.mark.validation
    @pytest.mark.blackjack
    def test_card_counting_accuracy_validation(self, blackjack_odds_calculator):
        """Validate card counting accuracy"""
        calc = blackjack_odds_calculator

        # Test with known card sequences
        test_sequences = [
            {
                'name': 'high_count_sequence',
                'cards': [
                    BlackjackCard('A', 'hearts', 11), BlackjackCard('K', 'spades', 10),
                    BlackjackCard('Q', 'diamonds', 10), BlackjackCard('J', 'clubs', 10),
                    BlackjackCard('10', 'hearts', 10)  # All high cards
                ],
                'expected_count_positive': True
            },
            {
                'name': 'low_count_sequence',
                'cards': [
                    BlackjackCard('2', 'hearts', 2), BlackjackCard('3', 'spades', 3),
                    BlackjackCard('4', 'diamonds', 4), BlackjackCard('5', 'clubs', 5),
                    BlackjackCard('6', 'hearts', 6)  # All low cards
                ],
                'expected_count_positive': False
            }
        ]

        validation_results = []

        for sequence in test_sequences:
            # Reset count
            calc.reset_count()

            # Update with test sequence
            calc.update_count(sequence['cards'])

            true_count = calc.calculate_true_count()

            result = {
                'scenario': sequence['name'],
                'running_count': calc.running_count,
                'true_count': true_count,
                'expected_positive': sequence['expected_count_positive'],
                'correct_sign': (true_count > 0) == sequence['expected_count_positive'],
                'timestamp': time.time()
            }

            validation_results.append(result)

            # Validate count sign is correct
            if sequence['expected_count_positive']:
                assert true_count > 0, f"Expected positive count for {sequence['name']}, got {true_count}"
            else:
                assert true_count < 0, f"Expected negative count for {sequence['name']}, got {true_count}"

        print("Card counting validation completed successfully")

    @pytest.mark.validation
    def test_cross_component_validation(self, sample_config, poker_odds_calculator, blackjack_odds_calculator):
        """Test that all components work together correctly"""
        # Test poker component
        poker_state = TestUtils.create_mock_game_state(
            'poker',
            player_cards=[Card('A', 'hearts'), Card('K', 'spades')],
            community_cards=[Card('Q', 'hearts'), Card('J', 'hearts'), Card('10', 'hearts')],
            pot_size=100,
            current_bet=20
        )

        poker_odds = poker_odds_calculator.calculate_odds(poker_state)
        assert poker_odds is not None
        assert poker_odds.equity_vs_random > 0
        assert poker_odds.recommended_action in ['fold', 'call', 'raise']

        # Test blackjack component
        blackjack_state = TestUtils.create_mock_game_state(
            'blackjack',
            player_cards=[BlackjackCard('A', 'hearts', 11), BlackjackCard('K', 'spades', 10)],
            dealer_upcard=BlackjackCard('6', 'diamonds', 6),
            player_score=21
        )

        blackjack_odds = blackjack_odds_calculator.calculate_odds(blackjack_state)
        assert blackjack_odds is not None
        assert blackjack_odds.player_win_probability > 0
        assert blackjack_odds.recommended_action in ['hit', 'stand', 'double', 'split', 'surrender']

        # Test database integration
        db = DatabaseManager(':memory:')

        poker_result = HandResult(
            game_type='poker',
            timestamp=time.time(),
            predicted_action=poker_odds.recommended_action,
            actual_outcome=poker_odds.recommended_action,  # Assume correct for this test
            expected_value=poker_odds.expected_value,
            actual_profit_loss=10.0,
            game_state_summary={'test': 'poker'},
            confidence_score=0.8
        )

        blackjack_result = HandResult(
            game_type='blackjack',
            timestamp=time.time(),
            predicted_action=blackjack_odds.recommended_action,
            actual_outcome=blackjack_odds.recommended_action,  # Assume correct for this test
            expected_value=blackjack_odds.player_win_probability,
            actual_profit_loss=10.0,
            game_state_summary={'test': 'blackjack'},
            confidence_score=0.8
        )

        # Test database operations
        poker_id = db.insert_hand_result(poker_result)
        blackjack_id = db.insert_hand_result(blackjack_result)

        assert poker_id > 0
        assert blackjack_id > 0

        # Test performance retrieval
        poker_perf = db.get_model_performance('poker')
        blackjack_perf = db.get_model_performance('blackjack')

        assert poker_perf['total_hands'] >= 1
        assert blackjack_perf['total_hands'] >= 1

        print("Cross-component validation completed successfully")


class TestContinuousValidation:
    """Continuous validation system for production monitoring"""

    def __init__(self):
        self.validation_log = []
        self.performance_baseline = {}

    def log_validation_result(self, test_name: str, result: Dict[str, Any]):
        """Log a validation result for continuous monitoring"""
        log_entry = {
            'test_name': test_name,
            'timestamp': time.time(),
            'result': result
        }
        self.validation_log.append(log_entry)

        # Keep only recent entries (last 1000)
        if len(self.validation_log) > 1000:
            self.validation_log = self.validation_log[-1000:]

    def check_performance_degradation(self) -> Dict[str, Any]:
        """Check for performance degradation over time"""
        if len(self.validation_log) < 10:
            return {'status': 'insufficient_data'}

        # Analyze recent performance vs baseline
        recent_entries = self.validation_log[-10:]

        issues = []
        for entry in recent_entries:
            test_name = entry['test_name']
            result = entry['result']

            # Check for accuracy degradation
            if 'accuracy' in result and result['accuracy'] < 0.8:
                issues.append(f"Low accuracy in {test_name}: {result['accuracy']:.2%}")

            # Check for performance degradation
            if 'avg_time' in result and result['avg_time'] > 0.1:  # > 100ms
                issues.append(f"Slow performance in {test_name}: {result['avg_time']:.3f}s")

        return {
            'status': 'degraded' if issues else 'healthy',
            'issues': issues,
            'recent_performance': recent_entries
        }

    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate a comprehensive validation report"""
        if not self.validation_log:
            return {'error': 'No validation data available'}

        # Calculate overall statistics
        total_tests = len(self.validation_log)
        recent_tests = self.validation_log[-100:] if len(self.validation_log) > 100 else self.validation_log

        # Group by test type
        test_types = {}
        for entry in recent_tests:
            test_name = entry['test_name']
            if test_name not in test_types:
                test_types[test_name] = []
            test_types[test_name].append(entry['result'])

        # Calculate averages
        averages = {}
        for test_name, results in test_types.items():
            if results and 'accuracy' in results[0]:
                avg_accuracy = sum(r['accuracy'] for r in results) / len(results)
                averages[test_name] = {'avg_accuracy': avg_accuracy}

        return {
            'total_validation_tests': total_tests,
            'recent_tests_analyzed': len(recent_tests),
            'test_type_averages': averages,
            'performance_status': self.check_performance_degradation(),
            'report_timestamp': time.time()
        }


# Global validation instance for continuous monitoring
continuous_validation = TestContinuousValidation()


@pytest.fixture
def validation_monitor():
    """Fixture that provides continuous validation monitoring"""
    return continuous_validation