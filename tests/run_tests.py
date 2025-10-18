#!/usr/bin/env python3
"""
Test runner script for AI Blackjack Poker Assistant
Provides comprehensive testing workflow with performance tracking
"""
import subprocess
import sys
import os
import time
import json
from typing import Dict, List, Any
from pathlib import Path


class TestRunner:
    """Comprehensive test runner with performance tracking"""

    def __init__(self):
        self.test_results = {}
        self.start_time = time.time()
        self.performance_metrics = {}

    def run_unit_tests(self) -> Dict[str, Any]:
        """Run all unit tests"""
        print("🧪 Running unit tests...")

        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/unit/',
            '-v',
            '--tb=short',
            '--durations=10',
            '-m', 'unit'
        ], capture_output=True, text=True, cwd=os.getcwd())

        self.test_results['unit_tests'] = {
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': result.duration if hasattr(result, 'duration') else 0
        }

        print(f"Unit tests completed with return code: {result.returncode}")
        return self._parse_pytest_output(result.stdout)

    def run_validation_tests(self) -> Dict[str, Any]:
        """Run validation tests"""
        print("✅ Running validation tests...")

        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/validation/',
            '-v',
            '--tb=short',
            '-m', 'validation',
            '--durations=10'
        ], capture_output=True, text=True, cwd=os.getcwd())

        self.test_results['validation_tests'] = {
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': result.duration if hasattr(result, 'duration') else 0
        }

        print(f"Validation tests completed with return code: {result.returncode}")
        return self._parse_pytest_output(result.stdout)

    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("🔗 Running integration tests...")

        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            'tests/integration/',
            '-v',
            '--tb=short',
            '-m', 'integration',
            '--durations=10'
        ], capture_output=True, text=True, cwd=os.getcwd())

        self.test_results['integration_tests'] = {
            'return_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'duration': result.duration if hasattr(result, 'duration') else 0
        }

        print(f"Integration tests completed with return code: {result.returncode}")
        return self._parse_pytest_output(result.stdout)

    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        print("⚡ Running performance tests...")

        # Test core functionality performance
        performance_results = self._run_performance_benchmarks()

        self.test_results['performance_tests'] = {
            'results': performance_results,
            'timestamp': time.time()
        }

        return performance_results

    def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        print("🚀 Running complete test suite...")

        all_results = {}

        # Run test categories
        all_results['unit'] = self.run_unit_tests()
        all_results['validation'] = self.run_validation_tests()
        all_results['integration'] = self.run_integration_tests()
        all_results['performance'] = self.run_performance_tests()

        # Calculate overall statistics
        total_tests = sum(r.get('tests', 0) for r in all_results.values() if isinstance(r, dict))
        total_passed = sum(r.get('passed', 0) for r in all_results.values() if isinstance(r, dict))
        total_failed = sum(r.get('failed', 0) for r in all_results.values() if isinstance(r, dict))

        overall_results = {
            'total_tests': total_tests,
            'passed': total_passed,
            'failed': total_failed,
            'success_rate': total_passed / total_tests if total_tests > 0 else 0,
            'duration': time.time() - self.start_time,
            'timestamp': time.time()
        }

        print("📊 Test Results Summary:")
        print(f"   Total Tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Failed: {total_failed}")
        print(f"   Success Rate: {overall_results['success_rate']:.2%}")
        print(f"   Duration: {overall_results['duration']:.2f}s")

        # Save detailed results
        self._save_test_report(overall_results, all_results)

        return overall_results

    def _parse_pytest_output(self, output: str) -> Dict[str, Any]:
        """Parse pytest output to extract key metrics"""
        lines = output.split('\n')
        results = {
            'tests': 0,
            'passed': 0,
            'failed': 0,
            'errors': 0,
            'skipped': 0
        }

        for line in lines:
            if 'passed' in line.lower() and 'failed' in line.lower():
                # Parse summary line like "===== 15 passed, 3 failed in 2.34s ====="
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed,' and i > 0:
                        results['passed'] = int(parts[i-1])
                    elif part == 'failed' and i > 0:
                        results['failed'] = int(parts[i-1])

        results['tests'] = results['passed'] + results['failed']
        results['success_rate'] = results['passed'] / results['tests'] if results['tests'] > 0 else 0

        return results

    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks"""
        print("   Running blackjack performance benchmark...")

        try:
            # Import and run blackjack performance test
            sys.path.append(os.path.join(os.getcwd(), 'src'))
            from blackjack_odds import BlackjackOddsCalculator, BlackjackCard, BlackjackGameState

            calc = BlackjackOddsCalculator()

            # Time multiple calculations
            start_time = time.time()
            iterations = 100

            for _ in range(iterations):
                state = BlackjackGameState(
                    player_cards=[BlackjackCard('A', 'hearts', 11), BlackjackCard('K', 'spades', 10)],
                    dealer_cards=[BlackjackCard('6', 'diamonds', 6)],
                    player_score=21,
                    dealer_upcard=BlackjackCard('6', 'diamonds', 6)
                )
                calc.calculate_odds(state)

            blackjack_time = time.time() - start_time
            blackjack_avg = blackjack_time / iterations

            print("   Running poker performance benchmark...")

            from poker_odds import PokerOddsCalculator, Card, PokerGameState

            poker_calc = PokerOddsCalculator()

            start_time = time.time()

            for _ in range(iterations):
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
            poker_avg = poker_time / iterations

            return {
                'blackjack_total_time': blackjack_time,
                'blackjack_avg_time': blackjack_avg,
                'poker_total_time': poker_time,
                'poker_avg_time': poker_avg,
                'iterations': iterations,
                'performance_acceptable': blackjack_avg < 0.1 and poker_avg < 0.1  # < 100ms per calculation
            }

        except Exception as e:
            print(f"Performance benchmark failed: {e}")
            return {'error': str(e)}

    def _save_test_report(self, overall_results: Dict[str, Any], detailed_results: Dict[str, Any]):
        """Save comprehensive test report"""
        report = {
            'overall_results': overall_results,
            'detailed_results': detailed_results,
            'test_environment': {
                'python_version': sys.version,
                'platform': sys.platform,
                'timestamp': time.time()
            },
            'performance_metrics': self.performance_metrics
        }

        # Save to file
        report_file = f"test_report_{int(time.time())}.json"
        report_path = os.path.join(os.getcwd(), 'tests', 'results', report_file)

        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"📄 Test report saved to: {report_path}")

    def generate_coverage_report(self) -> None:
        """Generate coverage report"""
        print("📊 Generating coverage report...")

        try:
            result = subprocess.run([
                sys.executable, '-m', 'coverage', 'run',
                '--source=src',
                '--omit=*/tests/*,*/test_*',
                '-m', 'pytest', 'tests/unit/', 'tests/validation/',
                '--tb=short'
            ], capture_output=True, text=True, cwd=os.getcwd())

            if result.returncode == 0:
                # Generate HTML report
                subprocess.run([
                    sys.executable, '-m', 'coverage', 'html',
                    '-d', 'htmlcov'
                ], cwd=os.getcwd())

                subprocess.run([
                    sys.executable, '-m', 'coverage', 'report',
                    '--show-missing'
                ], cwd=os.getcwd())

                print("✅ Coverage report generated in htmlcov/")
            else:
                print(f"❌ Coverage report generation failed: {result.stderr}")

        except Exception as e:
            print(f"❌ Coverage tool not available: {e}")


def main():
    """Main test runner function"""
    print("🤖 AI Blackjack Poker Assistant - Test Suite")
    print("=" * 50)

    runner = TestRunner()

    # Check if specific test type requested
    if len(sys.argv) > 1:
        test_type = sys.argv[1]

        if test_type == 'unit':
            results = {'unit': runner.run_unit_tests()}
        elif test_type == 'validation':
            results = {'validation': runner.run_validation_tests()}
        elif test_type == 'integration':
            results = {'integration': runner.run_integration_tests()}
        elif test_type == 'performance':
            results = {'performance': runner.run_performance_tests()}
        elif test_type == 'coverage':
            runner.generate_coverage_report()
            return
        else:
            print(f"❌ Unknown test type: {test_type}")
            print("Available types: unit, validation, integration, performance, coverage")
            return
    else:
        # Run full test suite
        results = runner.run_all_tests()

    # Exit with appropriate code
    if results.get('success_rate', 0) >= 0.8:  # 80% success rate
        print("🎉 Tests completed successfully!")
        sys.exit(0)
    else:
        print("❌ Tests completed with failures!")
        sys.exit(1)


if __name__ == '__main__':
    main()