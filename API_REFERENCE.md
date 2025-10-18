# 🔌 API Reference

## Developer API Documentation

This document provides comprehensive API documentation for developers integrating with or extending the AI Blackjack Poker Assistant.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Game State Objects](#game-state-objects)
3. [Analysis Results](#analysis-results)
4. [Configuration API](#configuration-api)
5. [Integration Examples](#integration-examples)
6. [Error Handling](#error-handling)

## Core Classes

### AIAgentManager

Main entry point for the AI assistant.

```python
class AIAgentManager:
    """Main manager class for the AI Blackjack Poker Assistant"""

    def __init__(self, config_path: str = None):
        """Initialize the AI agent manager

        Args:
            config_path: Path to configuration file (optional)
        """

    def start_agent(self) -> bool:
        """Start the AI agent

        Returns:
            bool: True if started successfully, False otherwise
        """

    def stop_agent(self) -> None:
        """Stop the AI agent gracefully"""

    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status

        Returns:
            Dict containing status information:
            {
                'running': bool,
                'enabled': bool,
                'paused': bool,
                'current_game': str,
                'detection_confidence': float,
                'performance_fps': float,
                'memory_usage_mb': float,
                'uptime_seconds': float
            }
        """

    def calibrate_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Calibrate game detection regions

        Returns:
            Dict mapping region names to (x, y, width, height) tuples
        """
```

### PokerBlackjackAIAgent

Core AI agent implementation.

```python
class PokerBlackjackAIAgent:
    """Main AI agent for poker and blackjack analysis"""

    def __init__(self, config_obj: Config):
        """Initialize the AI agent

        Args:
            config_obj: Configuration object with game and UI settings
        """

    def start(self) -> bool:
        """Start the agent analysis loop"""

    def stop(self) -> None:
        """Stop the agent"""

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information

        Returns:
            Dict with system, configuration, and component status
        """

    def save_debug_frame(self, filename: str = "debug_frame.png") -> None:
        """Save current frame for debugging"""

    def get_betting_context(self) -> Dict[str, Any]:
        """Get current betting context and analysis"""
```

## Game State Objects

### PokerGameState

Represents the current state of a poker game.

```python
@dataclass
class PokerGameState:
    """Poker game state information"""

    player_cards: List[Card]           # Player's hole cards
    community_cards: List[Card]        # Community cards (flop, turn, river)
    pot_size: float                    # Current pot size
    current_bet: float                 # Current bet to call
    player_stack: float                # Player's remaining stack
    player_position: str               # Player's table position (BTN, CO, etc.)
    player_count: int                  # Number of players at table
    opponent_stacks: Dict[str, float]  # Opponent stack sizes
    game_phase: str                    # preflop, flop, turn, river
    player_tendencies: Dict[str, str]  # Known opponent tendencies
    confidence: float                  # Detection confidence (0-1)
```

### BlackjackGameState

Represents the current state of a blackjack game.

```python
@dataclass
class BlackjackGameState:
    """Blackjack game state information"""

    player_cards: List[BlackjackCard]  # Player's cards
    dealer_cards: List[BlackjackCard]  # Dealer's cards
    player_score: int                  # Player's hand value
    dealer_upcard: BlackjackCard       # Dealer's visible card
    current_bet: float                 # Current bet amount
    player_bankroll: float             # Player's total bankroll
    deck_count: int                    # Number of decks in shoe
    true_count: float                  # Current true count
    deck_penetration: float            # How much of deck has been played
    player_can_double: bool            # Whether double down is available
    player_can_split: bool             # Whether split is available
    game_phase: str                    # dealing, player_turn, dealer_turn
    confidence: float                  # Detection confidence (0-1)
```

## Analysis Results

### PokerOdds

Poker analysis results and recommendations.

```python
@dataclass
class PokerOdds:
    """Poker odds calculation results"""

    hand_strength: float              # 0-1, probability of having best hand
    hand_rank: str                   # high_card, pair, two_pair, etc.
    hand_category: str               # monster, strong, medium, weak, trash
    equity_vs_random: float          # Equity vs random opponent
    equity_vs_range: float           # Equity vs opponent's range
    pot_odds: float                  # Current pot odds
    recommended_action: str          # fold, call, raise, all_in
    outs: int                        # Number of outs to improve
    drawing_to: List[str]            # What you're drawing to

    # Advanced analysis
    specific_hand_odds: Dict[str, float]  # Odds of making specific hands
    expected_value: float            # Expected value of decision
    opponent_range_analysis: Dict[str, float]  # Opponent range analysis
    position_advantage: float        # Positional advantage factor
    bluff_factor: float              # Likelihood opponent is bluffing
    fold_equity: float               # Equity from opponent folds
```

### BlackjackOdds

Blackjack analysis results and recommendations.

```python
@dataclass
class BlackjackOdds:
    """Blackjack odds calculation results"""

    player_win_probability: float    # Probability of player winning
    dealer_bust_probability: float   # Probability dealer will bust
    player_bust_probability: float   # Probability player will bust if hitting
    surrender_value: float           # Expected value of surrendering
    hit_value: float                 # Expected value of hitting
    stand_value: float               # Expected value of standing
    double_value: float              # Expected value of doubling
    split_value: float               # Expected value of splitting
    recommended_action: str          # hit, stand, double, split, surrender
    true_count: float                # Current true count
    deck_penetration: float          # Deck penetration percentage

    # Advanced analysis
    blackjack_probability: float     # Probability of getting blackjack
    push_probability: float          # Probability of tie
    face_card_probability: float     # Probability of face card next
    ace_probability: float           # Probability of ace next
    ten_rich_probability: float      # Probability deck favors player
    double_down_advantage: float     # When double down is advantageous
    split_advantage: float           # When splitting is advantageous
    insurance_value: float           # Expected value of insurance
    bankroll_risk: float             # Risk to bankroll

    # Strategy tracking
    strategy_deviation_applied: bool # Whether count-based deviation used
    strategy_deviation_reason: str   # Reason for deviation
    count_impact: float              # Impact of count on strategy
    running_count: int               # Current running count
    cards_seen: int                  # Total cards observed
    decks_remaining: float           # Estimated decks remaining
```

## Configuration API

### Configuration Management

```python
class ConfigManager:
    """Manages application configuration"""

    def __init__(self, config_file: str = "config/config.py"):
        """Initialize configuration manager"""

    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration"""

    def update_config(self, section: str, key: str, value: Any) -> None:
        """Update configuration value"""

    def save_config(self) -> None:
        """Save configuration to file"""

    def reload_config(self) -> None:
        """Reload configuration from file"""
```

### Game Rules Configuration

```python
# Blackjack rules configuration
blackjack_rules = {
    'decks': 6,
    'dealer_hits_soft_17': True,      # H17 vs S17
    'double_after_split': True,       # DAS allowed
    'surrender_allowed': True,        # Late surrender
    'blackjack_payout': 1.5,          # Blackjack payout
    'penetration_warning': 0.75       # Deck penetration alert
}

# Poker rules configuration
poker_rules = {
    'game_type': 'texas_holdem',
    'stakes': 'NLHE',
    'small_blind': 1.0,
    'big_blind': 2.0,
    'ante': 0.0,
    'max_buyin': 100.0,
    'player_count': 6
}
```

## Integration Examples

### Basic Integration

```python
from src.ai_agent import AIAgentManager
from src.poker_odds import PokerOddsCalculator
from src.blackjack_odds import BlackjackOddsCalculator

# Initialize the AI assistant
manager = AIAgentManager()

# Start the agent
if manager.start_agent():
    print("AI Assistant started successfully")
else:
    print("Failed to start AI Assistant")
    exit(1)

# Get current status
status = manager.get_agent_status()
print(f"Agent status: {status}")

# Your game loop here...
while True:
    # Get current game analysis
    diagnostics = manager.get_diagnostic_info()

    if diagnostics['agent_status']['current_game'] == 'poker':
        # Handle poker analysis
        betting_context = manager.agent.get_betting_context()
        print(f"Poker context: {betting_context}")
    elif diagnostics['agent_status']['current_game'] == 'blackjack':
        # Handle blackjack analysis
        betting_context = manager.agent.get_betting_context()
        print(f"Blackjack context: {betting_context}")

    # Sleep or wait for next iteration
    time.sleep(0.1)

# Graceful shutdown
manager.stop_agent()
```

### Custom Analysis Integration

```python
# Advanced integration with custom analysis
from src.screen_capture import ScreenCaptureManager
from src.poker_detector import PokerDetector
from src.blackjack_detector import BlackjackDetector

# Initialize components separately for custom integration
screen_manager = ScreenCaptureManager()
poker_detector = PokerDetector()
blackjack_detector = BlackjackDetector()
poker_odds = PokerOddsCalculator()
blackjack_odds = BlackjackOddsCalculator()

# Start screen capture
screen_manager.start_monitoring()

# Custom analysis loop
while True:
    # Get current frame
    frame = screen_manager.get_current_frame()
    if frame is None:
        time.sleep(0.1)
        continue

    # Detect game type and state
    poker_state = poker_detector.detect_game_state(frame)
    blackjack_state = blackjack_detector.detect_game_state(frame)

    # Analyze detected games
    if poker_state and poker_state.confidence > 0.8:
        odds = poker_odds.calculate_odds(poker_state)
        if odds:
            print(f"Poker odds: {odds.recommended_action}, {odds.hand_category}")

    if blackjack_state and blackjack_state.confidence > 0.8:
        odds = blackjack_odds.calculate_odds(blackjack_state)
        if odds:
            print(f"Blackjack odds: {odds.recommended_action}, Win: {odds.player_win_probability:.1%}")

    time.sleep(0.05)  # 20 FPS
```

### Web API Integration

```python
# Flask API for web integration
from flask import Flask, jsonify, request
from src.ai_agent import AIAgentManager

app = Flask(__name__)
manager = AIAgentManager()

@app.route('/api/status')
def api_status():
    """Get current agent status"""
    return jsonify(manager.get_agent_status())

@app.route('/api/analysis')
def api_analysis():
    """Get current game analysis"""
    return jsonify(manager.get_diagnostic_info())

@app.route('/api/blackjack/odds', methods=['POST'])
def api_blackjack_odds():
    """Calculate blackjack odds for provided game state"""
    game_state_data = request.get_json()
    # Convert to BlackjackGameState object
    # Calculate odds
    # Return results
    return jsonify({'status': 'not_implemented'})

@app.route('/api/poker/odds', methods=['POST'])
def api_poker_odds():
    """Calculate poker odds for provided game state"""
    game_state_data = request.get_json()
    # Convert to PokerGameState object
    # Calculate odds
    # Return results
    return jsonify({'status': 'not_implemented'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
```

## Error Handling

### Exception Hierarchy

```python
class PokerAIError(Exception):
    """Base exception for poker AI errors"""
    pass

class VisionError(PokerAIError):
    """Errors related to computer vision"""
    pass

class GameDetectionError(VisionError):
    """Errors in game state detection"""
    pass

class StrategyError(PokerAIError):
    """Errors in strategy calculation"""
    pass

class ConfigurationError(PokerAIError):
    """Errors in configuration"""
    pass

class PerformanceError(PokerAIError):
    """Performance-related errors"""
    pass
```

### Error Handling Best Practices

```python
from src.ai_agent import AIAgentManager
from src.exceptions import PokerAIError, VisionError

def safe_game_analysis():
    """Example of robust error handling"""

    manager = AIAgentManager()

    try:
        # Start agent with error handling
        if not manager.start_agent():
            raise PokerAIError("Failed to start AI agent")

        # Main analysis loop with error recovery
        while True:
            try:
                # Get current analysis
                status = manager.get_agent_status()

                if status['running']:
                    # Perform analysis
                    diagnostics = manager.get_diagnostic_info()
                    process_game_analysis(diagnostics)
                else:
                    # Agent not running, attempt restart
                    logger.warning("Agent not running, attempting restart")
                    if manager.start_agent():
                        logger.info("Agent restarted successfully")

            except VisionError as e:
                # Vision errors - may be temporary
                logger.warning(f"Vision error: {e}")
                time.sleep(1)  # Brief pause before retry

            except StrategyError as e:
                # Strategy errors - may indicate configuration issue
                logger.error(f"Strategy error: {e}")
                # Continue running but log for review

            except Exception as e:
                # Unexpected errors - log and continue
                logger.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(0.5)

    except PokerAIError as e:
        logger.critical(f"Critical poker AI error: {e}")
        return False

    except KeyboardInterrupt:
        logger.info("Received shutdown signal")

    except Exception as e:
        logger.critical(f"Unexpected critical error: {e}", exc_info=True)
        return False

    finally:
        # Always cleanup
        manager.stop_agent()
        logger.info("Agent stopped")

    return True
```

## Performance Monitoring API

### Metrics Collection

```python
class PerformanceMonitor:
    """API for performance monitoring"""

    def get_system_metrics(self) -> Dict[str, float]:
        """Get current system performance metrics

        Returns:
            Dict with CPU, memory, disk usage, etc.
        """

    def get_agent_metrics(self) -> Dict[str, float]:
        """Get AI agent performance metrics

        Returns:
            Dict with FPS, latency, accuracy metrics
        """

    def get_game_metrics(self) -> Dict[str, Any]:
        """Get game-specific performance metrics

        Returns:
            Dict with detection rates, analysis times, etc.
        """

    def export_metrics(self, format: str = 'json') -> str:
        """Export all metrics in specified format"""
```

## Integration Patterns

### Event-Driven Integration

```python
# Observer pattern for game events
class GameEventListener:
    """Listen for game events and trigger actions"""

    def on_game_detected(self, game_type: str, confidence: float):
        """Called when a game is detected"""
        pass

    def on_hand_updated(self, game_type: str, game_state: Any):
        """Called when hand state changes"""
        pass

    def on_analysis_complete(self, game_type: str, odds: Any):
        """Called when analysis is complete"""
        pass

    def on_error(self, error_type: str, error_details: Dict):
        """Called when errors occur"""
        pass
```

### Plugin Architecture

```python
# Plugin interface for extensibility
class GamePlugin:
    """Base class for game-specific plugins"""

    def __init__(self, name: str, version: str):
        self.name = name
        self.version = version

    def initialize(self, config: Dict) -> bool:
        """Initialize plugin with configuration"""
        return True

    def process_game_state(self, game_state: Any) -> Any:
        """Process game state and return enhanced results"""
        return game_state

    def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
```

## Testing API

### Unit Testing Support

```python
# Test utilities for integration testing
class TestUtilities:
    """Utilities for testing AI agent components"""

    @staticmethod
    def create_mock_poker_state(hand: List[str] = None,
                              board: List[str] = None) -> PokerGameState:
        """Create mock poker game state for testing"""

    @staticmethod
    def create_mock_blackjack_state(hand: List[str] = None,
                                  dealer: str = None) -> BlackjackGameState:
        """Create mock blackjack game state for testing"""

    @staticmethod
    def validate_odds_result(odds: Any) -> bool:
        """Validate that odds calculation results are reasonable"""

    @staticmethod
    def benchmark_performance(func: Callable, iterations: int = 100) -> Dict[str, float]:
        """Benchmark function performance"""
```

## Best Practices

### API Usage Guidelines

1. **Resource Management**
   ```python
   # Always use context managers for proper cleanup
   manager = AIAgentManager()
   try:
       manager.start_agent()
       # Use the agent
   finally:
       manager.stop_agent()
   ```

2. **Error Handling**
   ```python
   # Implement comprehensive error handling
   try:
       result = manager.get_diagnostic_info()
   except Exception as e:
       logger.error(f"API error: {e}")
       # Implement fallback behavior
   ```

3. **Performance Considerations**
   ```python
   # Cache expensive operations
   @lru_cache(maxsize=1000)
   def get_analysis_cached(game_state_hash):
       return manager.get_diagnostic_info()
   ```

4. **Configuration Management**
   ```python
   # Use configuration API for settings
   config = ConfigManager()
   config.update_config('performance', 'max_simulation_count', 5000)
   config.save_config()
   ```

## API Versioning

### Version Compatibility

| API Version | Status | Compatibility |
|-------------|--------|---------------|
| **v1.0.0** | Current | Fully backward compatible |
| **v0.9.0** | Deprecated | Use v1.0.0 |
| **v0.8.0** | Legacy | Limited support |

### Migration Guide

```python
# Migration from v0.9.0 to v1.0.0
def migrate_from_v090_to_v100():
    """Migration guide for API changes"""

    # Configuration changes
    old_config = load_old_config()
    new_config = migrate_config_structure(old_config)

    # API response changes
    old_response = call_old_api()
    new_response = migrate_response_format(old_response)

    return new_config, new_response
```

## Support and Troubleshooting

### Debug API

```python
# Debug utilities for troubleshooting
class DebugAPI:
    """API for debugging and troubleshooting"""

    def enable_debug_mode(self) -> None:
        """Enable detailed logging and debugging"""

    def save_debug_screenshot(self, filename: str) -> None:
        """Save current frame for analysis"""

    def get_internal_state(self) -> Dict[str, Any]:
        """Get internal component states"""

    def run_diagnostics(self) -> Dict[str, Any]:
        """Run comprehensive diagnostics"""
```

### Logging API

```python
# Logging configuration for integration
logging_config = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'handlers': {
        'file': {'filename': 'poker_ai.log', 'max_size': '10MB'},
        'console': {'enabled': True}
    }
}
```

This API reference provides comprehensive documentation for developers working with the AI Blackjack Poker Assistant. For additional support, please refer to the project documentation or community resources.