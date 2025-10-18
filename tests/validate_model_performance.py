#!/usr/bin/env python3
"""
Model Performance Validation Workflow
Tracks actual vs expected results for continuous model improvement
"""
import time
import json
import argparse
from typing import Dict, List, Any
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from blackjack_odds import BlackjackOddsCalculator, BlackjackCard, BlackjackGameState
from poker_odds import PokerOddsCalculator, Card, PokerGameState
from historical_tracker import DatabaseManager, HandResult


class ModelPerformanceTracker:
    """Tracks model performance over time for continuous validation"""

    def __init__(self, db_path: str = 'ai_assistant_validation.db'):
        self.db_manager = DatabaseManager(db_path)
        self.validation_log = []
        self.baseline_performance = {}

    def run_blackjack_validation(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run blackjack strategy validation"""
        print("🎯 Running Blackjack Strategy Validation...")

        calc = BlackjackOddsCalculator()
        results = []

        for scenario in scenarios:
            # Create game state
            game_state = BlackjackGameState(
                player_cards=scenario['player_cards'],
                dealer_cards=[scenario['dealer_card']],
                player_score=sum(card.value for card in scenario['player_cards']),
                dealer_upcard=scenario['dealer_card']
            )

            # Get prediction
            odds = calc.calculate_odds(game_state)

            if odds:
                prediction = odds.recommended_action
                expected = scenario['expected_action']

                result = {
                    'scenario_name': scenario['name'],
                    'prediction': prediction,
                    'expected': expected,
                    'correct': prediction == expected,
                    'confidence': odds.player_win_probability,
                    'true_count': odds.true_count,
                    'timestamp': time.time()
                }

                results.append(result)

                # Record in database
                hand_result = HandResult(
                    game_type='blackjack',
                    timestamp=time.time(),
                    predicted_action=prediction,
                    actual_outcome=expected,
                    expected_value=odds.player_win_probability,
                    actual_profit_loss=10 if prediction == expected else -10,
                    game_state_summary={
                        'player_score': game_state.player_score,
                        'dealer_upcard': game_state.dealer_upcard.value if game_state.dealer_upcard else 0
                    },
                    confidence_score=odds.player_win_probability
                )

                self.db_manager.insert_hand_result(hand_result)

        # Calculate accuracy
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / len(results) if results else 0

        validation_summary = {
            'test_type': 'blackjack_strategy',
            'total_scenarios': len(results),
            'correct_predictions': correct,
            'accuracy': accuracy,
            'results': results,
            'timestamp': time.time()
        }

        print(f"Blackjack validation: {accuracy:.2%} accuracy ({correct}/{len(results)})")

        return validation_summary

    def run_poker_validation(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run poker equity validation"""
        print("🃏 Running Poker Equity Validation...")

        calc = PokerOddsCalculator()
        results = []

        for scenario in scenarios:
            # Create game state
            game_state = PokerGameState(
                player_cards=scenario['player_cards'],
                community_cards=scenario.get('community_cards', []),
                pot_size=100,
                current_bet=20,
                total_bet=20,
                player_stack=500,
                player_position='BTN',
                player_count=6,
                opponent_stacks=[400]*5,
                game_phase='flop'
            )

            # Get prediction
            odds = calc.calculate_odds(game_state)

            if odds:
                equity = odds.equity_vs_random
                expected_min = scenario['expected_equity_min']

                result = {
                    'scenario_name': scenario['name'],
                    'equity': equity,
                    'expected_min': expected_min,
                    'correct': equity >= expected_min,
                    'hand_category': odds.hand_category,
                    'recommended_action': odds.recommended_action,
                    'timestamp': time.time()
                }

                results.append(result)

                # Record in database
                hand_result = HandResult(
                    game_type='poker',
                    timestamp=time.time(),
                    predicted_action=odds.recommended_action,
                    actual_outcome=odds.recommended_action,  # Assume correct for equity test
                    expected_value=equity,
                    actual_profit_loss=10 if equity >= expected_min else -10,
                    game_state_summary={
                        'hand_category': odds.hand_category,
                        'equity': equity
                    },
                    confidence_score=odds.hand_strength
                )

                self.db_manager.insert_hand_result(hand_result)

        # Calculate accuracy
        correct = sum(1 for r in results if r['correct'])
        accuracy = correct / len(results) if results else 0

        validation_summary = {
            'test_type': 'poker_equity',
            'total_scenarios': len(results),
            'correct_predictions': correct,
            'accuracy': accuracy,
            'results': results,
            'timestamp': time.time()
        }

        print(f"Poker validation: {accuracy:.2%} accuracy ({correct}/{len(results)})")

        return validation_summary

    def run_performance_validation(self) -> Dict[str, Any]:
        """Run performance validation tests"""
        print("⚡ Running Performance Validation...")

        # Blackjack performance test
        blackjack_calc = BlackjackOddsCalculator()
        start_time = time.time()

        for _ in range(50):
            state = BlackjackGameState(
                player_cards=[BlackjackCard('A', 'hearts', 11), BlackjackCard('K', 'spades', 10)],
                dealer_cards=[BlackjackCard('6', 'diamonds', 6)],
                player_score=21,
                dealer_upcard=BlackjackCard('6', 'diamonds', 6)
            )
            blackjack_calc.calculate_odds(state)

        blackjack_time = time.time() - start_time

        # Poker performance test
        poker_calc = PokerOddsCalculator()
        start_time = time.time()

        for _ in range(50):
            poker_state = PokerGameState(
                player_cards=[Card('A', 'hearts'), Card('K', 'spades')],
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
            poker_calc.calculate_odds(poker_state)

        poker_time = time.time() - start_time

        # Performance metrics
        performance_results = {
            'blackjack_avg_time': blackjack_time / 50,
            'poker_avg_time': poker_time / 50,
            'blackjack_total_time': blackjack_time,
            'poker_total_time': poker_time,
            'performance_acceptable': (blackjack_time / 50) < 0.1 and (poker_time / 50) < 0.1,
            'timestamp': time.time()
        }

        print(f"Performance validation: Blackjack {blackjack_time/50:.4f}s avg, Poker {poker_time/50:.4f}s avg")

        return performance_results

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        # Get historical performance from database
        blackjack_perf = self.db_manager.get_model_performance('blackjack')
        poker_perf = self.db_manager.get_model_performance('poker')

        # Get recent validation results
        recent_results = {
            'blackjack': blackjack_perf,
            'poker': poker_perf,
            'report_timestamp': time.time(),
            'validation_summary': {
                'total_blackjack_hands': blackjack_perf.get('total_hands', 0),
                'total_poker_hands': poker_perf.get('total_hands', 0),
                'overall_accuracy': (blackjack_perf.get('accuracy', 0) + poker_perf.get('accuracy', 0)) / 2,
                'blackjack_accuracy': blackjack_perf.get('accuracy', 0),
                'poker_accuracy': poker_perf.get('accuracy', 0)
            }
        }

        return recent_results

    def run_continuous_validation(self, interval_minutes: int = 60) -> None:
        """Run continuous validation at specified intervals"""
        print(f"🔄 Starting continuous validation (every {interval_minutes} minutes)...")

        try:
            while True:
                # Run validation scenarios
                blackjack_scenarios = [
                    {
                        'name': 'blackjack_21',
                        'player_cards': [BlackjackCard('A', 'hearts', 11), BlackjackCard('K', 'spades', 10)],
                        'dealer_card': BlackjackCard('6', 'diamonds', 6),
                        'expected_action': 'stand'
                    },
                    {
                        'name': 'blackjack_16',
                        'player_cards': [BlackjackCard('10', 'hearts', 10), BlackjackCard('6', 'spades', 6)],
                        'dealer_card': BlackjackCard('10', 'diamonds', 10),
                        'expected_action': 'hit'
                    }
                ]

                poker_scenarios = [
                    {
                        'name': 'pocket_aces',
                        'player_cards': [Card('A', 'hearts'), Card('A', 'spades')],
                        'expected_equity_min': 0.75
                    },
                    {
                        'name': 'ace_king',
                        'player_cards': [Card('A', 'hearts'), Card('K', 'spades')],
                        'expected_equity_min': 0.60
                    }
                ]

                # Run validations
                blackjack_results = self.run_blackjack_validation(blackjack_scenarios)
                poker_results = self.run_poker_validation(poker_scenarios)
                performance_results = self.run_performance_validation()

                # Log results
                self.validation_log.append({
                    'timestamp': time.time(),
                    'blackjack': blackjack_results,
                    'poker': poker_results,
                    'performance': performance_results
                })

                # Generate and save report
                report = self.generate_performance_report()
                self._save_continuous_report(report)

                print(f"⏰ Validation cycle completed. Next run in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            print("🛑 Continuous validation stopped by user")
        except Exception as e:
            print(f"❌ Error in continuous validation: {e}")

    def _save_continuous_report(self, report: Dict[str, Any]):
        """Save continuous validation report"""
        try:
            filename = f"continuous_validation_report_{int(time.time())}.json"
            filepath = os.path.join(os.getcwd(), 'tests', 'results', filename)

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        except Exception as e:
            print(f"Warning: Could not save continuous report: {e}")

    def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        if len(self.validation_log) < 2:
            return {'error': 'Insufficient data for trend analysis'}

        recent_logs = self.validation_log[-10:]  # Last 10 validation cycles

        # Extract accuracy trends
        blackjack_accuracies = [log['blackjack']['accuracy'] for log in recent_logs]
        poker_accuracies = [log['poker']['accuracy'] for log in recent_logs]

        # Calculate trends
        blackjack_trend = blackjack_accuracies[-1] - blackjack_accuracies[0] if len(blackjack_accuracies) > 1 else 0
        poker_trend = poker_accuracies[-1] - poker_accuracies[0] if len(poker_accuracies) > 1 else 0

        return {
            'blackjack_accuracy_trend': blackjack_trend,
            'poker_accuracy_trend': poker_trend,
            'blackjack_current_accuracy': blackjack_accuracies[-1] if blackjack_accuracies else 0,
            'poker_current_accuracy': poker_accuracies[-1] if poker_accuracies else 0,
            'analysis_timestamp': time.time()
        }


def main():
    """Main validation workflow function"""
    parser = argparse.ArgumentParser(description='Model Performance Validation Workflow')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuous validation')
    parser.add_argument('--interval', type=int, default=60,
                       help='Interval for continuous validation (minutes)')
    parser.add_argument('--blackjack-only', action='store_true',
                       help='Run only blackjack validation')
    parser.add_argument('--poker-only', action='store_true',
                       help='Run only poker validation')
    parser.add_argument('--performance-only', action='store_true',
                       help='Run only performance validation')

    args = parser.parse_args()

    tracker = ModelPerformanceTracker()

    print("🎯 AI Assistant Model Performance Validation")
    print("=" * 50)

    # Define validation scenarios
    blackjack_scenarios = [
        {
            'name': 'blackjack_21_vs_6',
            'player_cards': [BlackjackCard('A', 'hearts', 11), BlackjackCard('K', 'spades', 10)],
            'dealer_card': BlackjackCard('6', 'diamonds', 6),
            'expected_action': 'stand'
        },
        {
            'name': 'blackjack_16_vs_10',
            'player_cards': [BlackjackCard('10', 'hearts', 10), BlackjackCard('6', 'spades', 6)],
            'dealer_card': BlackjackCard('10', 'diamonds', 10),
            'expected_action': 'hit'
        },
        {
            'name': 'blackjack_pair_aces',
            'player_cards': [BlackjackCard('A', 'hearts', 11), BlackjackCard('A', 'spades', 11)],
            'dealer_card': BlackjackCard('6', 'diamonds', 6),
            'expected_action': 'split'
        }
    ]

    poker_scenarios = [
        {
            'name': 'pocket_aces_preflop',
            'player_cards': [Card('A', 'hearts'), Card('A', 'spades')],
            'expected_equity_min': 0.75
        },
        {
            'name': 'ace_king_preflop',
            'player_cards': [Card('A', 'hearts'), Card('K', 'spades')],
            'expected_equity_min': 0.60
        }
    ]

    if args.performance_only:
        # Run only performance validation
        performance_results = tracker.run_performance_validation()
        print(f"Performance validation completed: {performance_results}")

    elif args.blackjack_only:
        # Run only blackjack validation
        blackjack_results = tracker.run_blackjack_validation(blackjack_scenarios)
        print(f"Blackjack validation completed: {blackjack_results['accuracy']:.2%} accuracy")

    elif args.poker_only:
        # Run only poker validation
        poker_results = tracker.run_poker_validation(poker_scenarios)
        print(f"Poker validation completed: {poker_results['accuracy']:.2%} accuracy")

    elif args.continuous:
        # Run continuous validation
        tracker.run_continuous_validation(args.interval)

    else:
        # Run all validations once
        print("Running complete validation suite...")

        blackjack_results = tracker.run_blackjack_validation(blackjack_scenarios)
        poker_results = tracker.run_poker_validation(poker_scenarios)
        performance_results = tracker.run_performance_validation()

        # Generate final report
        final_report = {
            'blackjack_validation': blackjack_results,
            'poker_validation': poker_results,
            'performance_validation': performance_results,
            'summary': {
                'blackjack_accuracy': blackjack_results['accuracy'],
                'poker_accuracy': poker_results['accuracy'],
                'performance_acceptable': performance_results['performance_acceptable'],
                'overall_score': (blackjack_results['accuracy'] + poker_results['accuracy']) / 2,
                'timestamp': time.time()
            }
        }

        # Save final report
        report_file = f"final_validation_report_{int(time.time())}.json"
        report_path = os.path.join(os.getcwd(), 'tests', 'results', report_file)

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        print("✅ Complete validation suite finished!")
        print(f"📊 Blackjack accuracy: {blackjack_results['accuracy']:.2%}")
        print(f"📊 Poker accuracy: {poker_results['accuracy']:.2%}")
        print(f"⚡ Performance acceptable: {performance_results['performance_acceptable']}")
        print(f"📄 Report saved to: {report_path}")


if __name__ == '__main__':
    main()