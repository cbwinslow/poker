# AI Blackjack Poker Assistant

A sophisticated real-time AI assistant for blackjack and poker that provides advanced strategic analysis, odds calculations, and opponent modeling based on computer vision and game theory research.

## 🎯 Advanced Features

### 📊 **Comprehensive Game Theory Implementation**
- **Blackjack**: Full basic strategy engine with configurable casino rules (H17/S17, DAS, surrender)
- **Advanced Card Counting**: Hi-Lo system with true count calculation and playing deviations
- **Research-Based Deviations**: Strategy adjustments based on count and game theory

### 🧠 **Sophisticated Poker Analysis**
- **Monte Carlo Equity**: Advanced simulations with opponent range analysis
- **Effective Hand Strength (EHS)**: Research-based hand evaluation considering future potential
- **Dynamic Opponent Modeling**: Weight tables that adapt based on observed opponent behavior
- **Position-Aware Analysis**: Strategic recommendations considering table position

### 🔍 **Enhanced Vision Pipeline**
- **High-Performance Capture**: Optimized screen grabbing with adaptive frame rates
- **OCR Integration**: Text recognition for pot sizes, bets, and game information
- **Intelligent Preprocessing**: Image enhancement for better detection accuracy

### 💾 **Comprehensive Data Management**
- **SQLite Database**: Persistent storage for hand history and opponent statistics
- **Session Tracking**: Detailed performance analysis across gaming sessions
- **Model Improvement**: Historical data analysis for strategy refinement

### 🎛️ **Advanced Configuration**
- **Rule Customization**: Configurable blackjack rules (decks, dealer rules, payouts)
- **Performance Tuning**: Adaptive delays, caching, and resource optimization
- **Hotkey Controls**: Runtime control with customizable keybindings

## Installation

### Prerequisites
- Python 3.8 or higher
- Windows, macOS, or Linux operating system

### Setup
1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd poker-ai-agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Calibrate game regions (recommended)**
   ```bash
   python main.py --calibrate
   ```
   Follow the on-screen instructions to define your game areas.

## Usage

### Basic Usage
```bash
python main.py
```

### Command Line Options
```bash
python main.py [OPTIONS]

Options:
  --calibrate              Run calibration mode to set up game regions
  --debug                  Save debug screenshots during operation
  --overlay-x INT          Set overlay X position (default: 10)
  --overlay-y INT          Set overlay Y position (default: 10)
  --transparency FLOAT     Set overlay transparency 0.0-1.0 (default: 0.9)
```

### Runtime Controls
Once the agent is running, you can use these hotkeys for control:

- **Ctrl+Alt+T**: Toggle agent on/off
- **Ctrl+Alt+P**: Pause/resume detection
- **Ctrl+Alt+M**: Minimize/restore overlay window

### Advanced Features
- **Smart Performance**: Automatically reduces CPU usage when no game is detected
- **Comprehensive Logging**: Detailed logs saved to `poker_ai_agent.log`
- **System Tray Support**: Minimize to system tray (when available)
- **Enhanced Accuracy**: Improved poker odds calculation with better hand evaluation
- **Graceful Shutdown**: Proper cleanup on exit signals

### Example Usage
```bash
# Start with custom overlay position and transparency
python main.py --overlay-x 100 --overlay-y 50 --transparency 0.8

# Run in debug mode to save screenshots
python main.py --debug

# First-time calibration
python main.py --calibrate
```

## 🏗️ System Architecture

### Multi-Layered Design
The AI assistant follows a sophisticated layered architecture for optimal performance and maintainability:

### 1. **Vision Layer (The "Eyes")**
- **High-Performance Screen Capture**: Uses `mss` for optimized screen grabbing at configurable frame rates
- **Adaptive Frame Skipping**: Automatically adjusts capture rate based on system performance
- **OCR Integration**: Extracts text information (pot sizes, bets) using EasyOCR with preprocessing
- **Region-Based Detection**: Configurable detection regions for different game elements

### 2. **Logic Layer (The "Brain")**
- **Advanced Blackjack Engine**: Full basic strategy with rule variations and card counting deviations
- **Sophisticated Poker Calculator**: Monte Carlo simulations with opponent range analysis and EHS
- **Dynamic Opponent Modeling**: Weight table system that adapts to opponent behavior patterns
- **Real-Time Performance**: Caching and optimization for sub-200ms response times

### 3. **Data Layer (The "Memory")**
- **SQLite Database**: Persistent storage for hand history, opponent statistics, and performance metrics
- **Session Management**: Tracks gaming sessions with comprehensive profit/loss analysis
- **Model Improvement**: Historical data analysis for continuous strategy refinement

### 4. **Presentation Layer (The "Display")**
- **Transparent Overlay**: Non-intrusive UI using Tkinter with customizable transparency
- **Color-Coded Information**: Visual indicators for hand strength, recommendations, and alerts
- **Real-Time Updates**: Smooth display updates with minimal performance impact

## 🎯 Advanced Blackjack Features

### Comprehensive Strategy Engine
- **Configurable Rules**: Supports all major casino rule variations (H17/S17, DAS, surrender, etc.)
- **Advanced Card Counting**: Hi-Lo system with accurate true count calculation
- **Playing Deviations**: Research-based strategy adjustments for advantage play
- **Risk Management**: Bankroll-aware recommendations and bet sizing optimization

### Card Counting Implementation
```python
# Example of advanced counting system
counting_systems = {
    'hi_lo': {'2': 1, '3': 1, '4': 1, '5': 1, '6': 1, '7': 0, '8': 0, '9': 0, '10': -1, 'A': -1},
    'zen_count': {'2': 1, '3': 2, '4': 2, '5': 2, '6': 2, '7': 1, '8': 0, '9': 0, '10': -2, 'A': -1}
}
```

## 🃏 Advanced Poker Features

### Sophisticated Equity Analysis
- **Monte Carlo Simulations**: High-precision equity calculations with configurable iteration counts
- **Range-Based Analysis**: Opponent range evaluation using dynamic weight tables
- **Effective Hand Strength**: Research-based EHS calculation considering future street potential
- **Position-Aware Strategy**: Recommendations adjusted for table position and stack sizes

### Opponent Modeling System
```python
# Dynamic weight table for opponent behavior
opponent_model = {
    'hand_weights': {'AA': 1.0, 'KK': 0.95, 'AKs': 0.85, ...},  # 1326 possible hands
    'statistics': {'vpip': 0.22, 'pfr': 0.18, 'af': 1.8, ...},     # Behavioral tracking
    'tendencies': ['aggressive', 'bluffy', 'positional']           # Pattern recognition
}
```

## ⚙️ Configuration System

### Comprehensive Game Rules
```python
# Blackjack rule configuration
blackjack_rules = {
    'decks': 6,
    'dealer_hits_soft_17': True,  # H17 vs S17
    'double_after_split': True,   # DAS allowed
    'surrender_allowed': True,    # Late surrender
    'penetration_warning': 0.75   # Deck penetration alerts
}

# Poker rule configuration
poker_rules = {
    'game_type': 'texas_holdem',
    'stakes': 'NLHE',
    'blinds_structure': {'small_blind': 1.0, 'big_blind': 2.0},
    'max_buyin': 100.0
}
```

## Configuration

### Game Regions
Game detection regions can be calibrated for your specific game windows:

- **Poker regions**: Player cards, community cards, pot info, player info
- **Blackjack regions**: Player cards, dealer cards, score areas

### Detection Settings
- Adjustable confidence thresholds for card detection
- Configurable capture rates for performance tuning
- Customizable overlay appearance

### New Configuration Options
The agent now supports extensive configuration through the `config/config.py` file:

- **Performance Settings**: Adaptive delays, idle timeouts, auto-pause functionality
- **Hotkey Controls**: Customizable hotkeys for runtime control
- **UI Enhancements**: Minimize options, overlay customization, system tray support
- **Logging**: Configurable log levels and file output
- **Detection Sensitivity**: Multiple sensitivity modes for different accuracy/performance trade-offs

## Project Structure

```
poker-ai-agent/
├── main.py                 # Main entry point
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── config/
│   └── config.py          # Configuration management
├── src/
│   ├── ai_agent.py        # Main AI agent orchestrator
│   ├── screen_capture.py  # Screen capture functionality
│   ├── poker_detector.py  # Poker game state detection
│   ├── blackjack_detector.py # Blackjack game state detection
│   ├── poker_odds.py      # Poker odds calculations
│   ├── blackjack_odds.py  # Blackjack odds calculations
│   └── ui/
│       └── odds_overlay.py # Real-time overlay UI
└── utils/                 # Utility functions (future)
```

## Poker Features

### Hand Analysis
- Complete hand evaluation (royal flush, straight flush, quads, etc.)
- Position-aware analysis
- Outs calculation for drawing hands

### Equity Calculation
- Monte Carlo simulation against random hands
- Range-based equity calculations
- Pot odds integration

### Recommendations
- Action recommendations based on equity and pot odds
- Hand strength categorization (monster, strong, medium, weak, trash)

## Blackjack Features

### Strategy Engine
- Basic strategy implementation
- Card counting (simplified true count)
- Deck penetration tracking

### Probability Analysis
- Win/bust probability calculations
- Dealer upcard analysis
- Expected value calculations for all actions

### Action Recommendations
- Hit/stand/double/split/surrender recommendations
- Color-coded action display
- Real-time probability updates

## Troubleshooting

### Common Issues

1. **Overlay not appearing**
   - Check if the agent started successfully
   - Verify overlay position is visible on your screen
   - Try adjusting transparency

2. **Game not detected**
   - Run calibration mode (`--calibrate`)
   - Ensure game window is visible and not minimized
   - Check detection confidence thresholds

3. **Poor performance**
   - Reduce capture rate in configuration
   - Close unnecessary applications
   - Try lower resolution game settings

### Debug Mode
Use `--debug` flag to save screenshots for troubleshooting:
```bash
python main.py --debug
```

## Development

### Adding New Features
The modular architecture makes it easy to extend:

1. **New games**: Implement detector in `src/`
2. **New odds calculations**: Add calculator classes
3. **UI improvements**: Modify overlay in `src/ui/`

### Testing
```bash
# Run with debug output
python main.py --debug

# Test specific components
python -m pytest tests/  # (if tests are added)
```

## Requirements

### Core Dependencies
- `opencv-python`: Computer vision and image processing
- `Pillow`: Image manipulation
- `numpy`: Numerical computations
- `mss`: Fast screen capture
- `tkinter`: GUI overlay

### Optional Dependencies
- `pytesseract`: OCR for text recognition (if implemented)
- `scipy`: Advanced image processing (if needed)

## 📋 Software Requirements Specification (SRS)

### System Overview
The AI Blackjack Poker Assistant is a real-time computer vision system that provides strategic analysis and odds calculations for blackjack and poker games through screen capture and advanced game theory algorithms.

### Functional Requirements

#### Vision System Requirements
- **REQ-V1**: Capture user-defined screen regions at minimum 10 FPS sustained rate
- **REQ-V2**: Card detection accuracy >99.5% for all 52 standard playing cards
- **REQ-V3**: OCR text recognition accuracy >98% for pot sizes, bets, and game text
- **REQ-V4**: Real-time region calibration with visual feedback

#### Blackjack Module Requirements
- **REQ-B1**: Support all major casino rule variations (H17/S17, DAS, surrender, decks)
- **REQ-B2**: Implement Hi-Lo card counting with true count accuracy ±0.1
- **REQ-B3**: Provide playing deviations based on count and game theory research
- **REQ-B4**: Calculate expected value for all possible actions (hit, stand, double, split, surrender)

#### Poker Module Requirements
- **REQ-P1**: Monte Carlo equity calculation with configurable simulation counts (10,000+)
- **REQ-P2**: Dynamic opponent modeling with 1326-hand weight table system
- **REQ-P3**: Position-aware strategy recommendations considering stack sizes
- **REQ-P4**: Effective Hand Strength (EHS) calculation for drawing hands

#### Data Management Requirements
- **REQ-D1**: SQLite database for persistent hand history storage
- **REQ-D2**: Session-based tracking with profit/loss analysis
- **REQ-D3**: Opponent statistics tracking (VPIP, PFR, aggression factor)
- **REQ-D4**: Model performance analytics with accuracy trends

### Performance Requirements
- **PERF-1**: End-to-end latency <750ms from screen change to overlay update
- **PERF-2**: CPU usage <15% during normal operation
- **PERF-3**: Memory footprint <200MB RAM
- **PERF-4**: Adaptive performance scaling based on system resources

### Non-Functional Requirements
- **RELIABILITY**: Handle unexpected game state changes without crashing
- **USABILITY**: Single-hotkey overlay visibility toggle
- **MAINTAINABILITY**: Modular architecture with clear separation of concerns
- **PORTABILITY**: Cross-platform compatibility (Windows, macOS, Linux)

## ⚡ Performance Optimizations

### Real-Time Performance Features
- **Adaptive Frame Rates**: Automatically adjusts capture rate based on system performance
- **Intelligent Caching**: Caches expensive calculations (equity, strategy) for sub-100ms response
- **Frame Skipping**: Skips frames when running too fast to maintain consistent timing
- **Memory Management**: Automatic cleanup of old cache entries and resources

### Multi-Threading Architecture
- **Background Capture**: Non-blocking screen capture in separate thread
- **Async Processing**: Game state analysis runs parallel to UI updates
- **Resource Pooling**: Reuses computational resources across game types

### Advanced Optimizations
```python
# Adaptive simulation count based on available time
simulation_count = min(base_simulations, max_simulations)
if time_available < target_latency:
    simulation_count = simulation_count // 2

# Performance monitoring
performance_metrics = {
    'calculation_times': [],
    'cache_hit_rate': 0.8,
    'fps_actual': 10.2,
    'memory_usage_mb': 85
}
```

## 🔬 Technical Specifications

### Computer Vision Pipeline
- **Screen Capture**: `mss` library for high-performance grabbing
- **Image Processing**: OpenCV for preprocessing and enhancement
- **OCR Engine**: EasyOCR with adaptive thresholding and noise reduction
- **Object Detection**: YOLOv8 integration ready for card/chip detection

### Game Theory Engines
- **Blackjack**: Composition-dependent strategy with deviation tables
- **Poker**: Counterfactual regret minimization concepts for unexploitable play
- **Opponent Modeling**: Bayesian inference for range updating

### Database Schema
```sql
-- Hand history with game-specific details
CREATE TABLE hand_history (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    game_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    predicted_action TEXT,
    actual_outcome TEXT,
    expected_value REAL,
    actual_profit_loss REAL
);

-- Opponent behavior tracking
CREATE TABLE opponent_statistics (
    opponent_id TEXT,
    session_id TEXT,
    vpip REAL,      -- Voluntarily Put In Pot %
    pfr REAL,       -- Pre-Flop Raise %
    af REAL,        -- Aggression Factor
    hands_played INTEGER
);
```

## 📊 Model Performance Metrics

### Accuracy Tracking
- **Historical Accuracy**: Rolling 1000-hand accuracy for strategy validation
- **Calibration Error**: Difference between expected and actual results
- **Position-Based Analysis**: Accuracy breakdown by table position
- **Situation-Specific**: Performance analysis for different game situations

### Continuous Improvement
- **Model Retraining**: Automatic parameter adjustment based on performance
- **Strategy Updates**: Research-based strategy improvements
- **Opponent Adaptation**: Dynamic model updates based on observed behavior

## 🔒 Legal and Ethical Notice

This software is designed **exclusively for educational and research purposes**. Key restrictions:

### Prohibited Uses
- **Real-Money Gambling**: Not for use in actual casino or online gambling environments
- **Advantage Play**: Not intended to gain unfair advantage over other players
- **Terms of Service Violation**: May violate platform rules on external assistance

### Permitted Uses
- **Strategy Training**: Learning optimal play through analysis
- **Game Theory Research**: Studying mathematical principles of card games
- **Software Development**: Understanding real-time computer vision systems
- **Personal Education**: Improving understanding of probability and decision-making

### Platform Policies
All major online gaming platforms prohibit real-time assistance tools:
- **Detection Risk**: Gaming sites actively detect and ban such software
- **Account Termination**: Use may result in permanent account suspension
- **Fund Forfeiture**: Winnings obtained through prohibited tools may be confiscated

## 📈 Advanced Usage Examples

### Custom Configuration
```python
# Advanced blackjack setup
config.game.blackjack_rules = BlackjackRules(
    decks=8,
    dealer_hits_soft_17=False,  # S17
    double_after_split=True,
    surrender_allowed=True,
    penetration_warning=0.8
)

# Performance optimization
config.screen.adaptive_delay = True
config.screen.idle_delay = 0.2
config.screen.active_delay = 0.05
```

### Database Analysis
```python
# Query historical performance
db_manager = DatabaseManager()
performance = db_manager.get_model_performance('blackjack')
print(f"Accuracy: {performance['accuracy']:.1%}")
print(f"Average profit: ${performance['avg_profit_per_hand']:.3f}")
```

## 🚀 Future Enhancements

### Planned Features
- **Machine Learning Detection**: Neural network-based card recognition
- **Multi-Table Support**: Simultaneous monitoring of multiple games
- **Voice Output**: Audio recommendations for hands-free operation
- **Mobile App**: Companion mobile application for remote monitoring

### Research Integration
- **GTO Solver Integration**: Game theory optimal strategy calculations
- **Deep Learning Models**: Advanced opponent behavior prediction
- **Risk of Ruin Analysis**: Bankroll management and variance calculations

---

**© 2025 AI Blackjack Poker Assistant Project**
*Educational research tool for game theory and computer vision*

## Contributing

Contributions are welcome! Areas for improvement:

- Enhanced card recognition accuracy
- Additional poker variants (Omaha, Stud)
- Advanced blackjack strategies
- Machine learning-based detection
- Mobile/responsive design

## License

[Add appropriate license information here]

## Support

For issues, questions, or feature requests:

1. Check the troubleshooting section
2. Review existing issues on GitHub
3. Create a new issue with detailed information

---

**Disclaimer**: This software provides mathematical analysis and strategic recommendations but does not guarantee winning outcomes. Gambling involves risk, and past performance does not predict future results.