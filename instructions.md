# 📋 Instructions Manual

## Complete Setup and Usage Guide

This comprehensive guide provides step-by-step instructions for setting up, configuring, and using the AI Blackjack Poker Assistant effectively.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Initial Configuration](#initial-configuration)
4. [Calibration Process](#calibration-process)
5. [Basic Usage](#basic-usage)
6. [Advanced Configuration](#advanced-configuration)
7. [Troubleshooting](#troubleshooting)
8. [Performance Optimization](#performance-optimization)
9. [Maintenance](#maintenance)

## System Requirements

### Minimum Specifications

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Operating System** | Windows 10, macOS 10.15, Ubuntu 20.04 | Windows 11, macOS 12+, Ubuntu 22.04 |
| **Processor** | Intel Core i5-8xxx, AMD Ryzen 5 | Intel Core i7-10xxx, AMD Ryzen 7 |
| **Memory** | 8 GB RAM | 16 GB RAM |
| **Graphics** | Integrated graphics | Dedicated GPU with 2GB+ VRAM |
| **Storage** | 500 MB free space | 1 GB free space |
| **Display** | 1920x1080 resolution | 2560x1440 resolution |
| **Network** | Stable internet connection | High-speed internet |

### Required Software

- **Python 3.8+**: Latest Python 3.x version
- **Visual C++ Redistributables** (Windows only)
- **Web browsers**: Chrome, Firefox, Safari, or Edge
- **Online gaming platforms**: PokerStars, 888poker, etc. (for testing)

## Installation

### Step 1: Download and Setup

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd poker-ai-agent
   ```

2. **Create virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   **Note**: If you encounter installation issues, install packages individually:
   ```bash
   pip install opencv-python numpy mss pyautogui
   pip install easyocr tesserocr torch ultralytics
   pip install PyQt6 pynput psutil
   ```

### Step 2: Verify Installation

1. **Test basic functionality**
   ```bash
   python -c "import cv2, numpy, mss, easyocr; print('Core dependencies OK')"
   ```

2. **Check screen capture**
   ```bash
   python -c "from src.screen_capture import ScreenCapture; print('Screen capture OK')"
   ```

3. **Verify OCR functionality**
   ```bash
   python -c "import easyocr; reader = easyocr.Reader(['en']); print('OCR OK')"
   ```

## Initial Configuration

### Configuration Files

The system uses a hierarchical configuration system:

```
config/
└── config.py          # Main configuration file
```

### Basic Configuration Settings

```python
# config/config.py - Key settings to modify

# Screen capture settings
screen_config = {
    'monitor': 0,                    # Primary monitor
    'capture_rate': 20,             # FPS for screen capture
    'adaptive_delay': True,         # Auto-adjust performance
    'region': None                  # Auto-detect or set manually
}

# Game detection settings
game_config = {
    'card_detection_threshold': 0.8,  # Minimum confidence for cards
    'ocr_confidence': 0.7,           # Minimum confidence for text
    'poker_regions': {},             # Will be set during calibration
    'blackjack_regions': {}          # Will be set during calibration
}

# UI settings
ui_config = {
    'overlay_alpha': 0.9,            # Transparency (0.0-1.0)
    'enable_hotkeys': True,         # Enable keyboard shortcuts
    'toggle_hotkey': 'ctrl+alt+t',   # Toggle agent on/off
    'pause_hotkey': 'ctrl+alt+p',    # Pause/resume detection
    'overlay_position': (10, 10)     # Overlay window position
}

# Performance settings
performance_config = {
    'max_simulation_count': 10000,   # Maximum Monte Carlo simulations
    'cache_enabled': True,           # Enable result caching
    'adaptive_performance': True,    # Auto-adjust based on system load
    'memory_limit_mb': 200           # Memory usage limit
}
```

### Environment Variables

Create a `.env` file for sensitive configuration:

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
DEBUG_MODE=true
LOG_LEVEL=INFO
```

## Calibration Process

### Why Calibration is Important

Calibration ensures accurate detection by mapping the AI's understanding to your specific:
- Screen resolution and layout
- Game window position and size
- Card positions and text locations
- Color schemes and themes

### Step-by-Step Calibration

1. **Start calibration mode**
   ```bash
   python main.py --calibrate
   ```

2. **Prepare your game**
   - Open your poker/blackjack game
   - Ensure game window is visible and not minimized
   - Set game to a standard table/view
   - Make sure cards are clearly visible

3. **Poker calibration process**
   ```
   Follow the prompts:
   1. Click on your hole cards (top-left corner, then bottom-right corner)
   2. Click on community cards area
   3. Click on pot size display
   4. Click on player information area
   5. Click on bet size indicators
   ```

4. **Blackjack calibration process**
   ```
   Follow the prompts:
   1. Click on player card area
   2. Click on dealer card area
   3. Click on score displays
   4. Click on bet amount area
   5. Click on bankroll/chip display
   ```

5. **Verification**
   - The system will display detected regions
   - Test with a few hands to ensure accuracy
   - Re-calibrate if detection accuracy is poor

### Calibration Troubleshooting

**Problem**: Cards not detected accurately
**Solution**:
- Ensure good lighting on screen
- Avoid screen glare or reflections
- Try different screen brightness settings
- Re-calibrate with larger click areas

**Problem**: Text (pot sizes, bets) not recognized
**Solution**:
- Ensure clear, readable text in game
- Try different text colors/sizes in game settings
- Check that game window isn't scaled

## Basic Usage

### Starting the Application

1. **Basic startup**
   ```bash
   python main.py
   ```

2. **With custom overlay position**
   ```bash
   python main.py --overlay-x 100 --overlay-y 50
   ```

3. **With custom transparency**
   ```bash
   python main.py --transparency 0.8
   ```

4. **Debug mode (saves screenshots)**
   ```bash
   python main.py --debug
   ```

### Runtime Operation

1. **Application startup**
   - The overlay window will appear (transparent)
   - Screen capture begins automatically
   - Game detection starts immediately

2. **Normal operation**
   - Play your game normally
   - AI analysis appears on overlay
   - Real-time odds and recommendations update

3. **Overlay information**
   ```
   ┌─────────────────────────────────────┐
   │ POKER ANALYSIS                      │
   │ Hand: A♠ K♦ (Suited Ace-King)      │
   │ Strength: Strong (68% vs random)    │
   │ Pot Odds: 3:1 (25% required)       │
   │ Recommended: RAISE                 │
   │ Position: BTN (Button)             │
   └─────────────────────────────────────┘
   ```

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+Alt+T** | Toggle agent on/off |
| **Ctrl+Alt+P** | Pause/resume detection |
| **Ctrl+Alt+M** | Minimize/restore overlay |
| **Ctrl+C** | Graceful shutdown |

## Advanced Configuration

### Game-Specific Settings

#### Poker Configuration

```python
# Advanced poker settings
poker_config = {
    'monte_carlo_simulations': 10000,      # Number of simulations
    'opponent_range_analysis': True,       # Enable range analysis
    'position_aware_strategy': True,       # Position-based recommendations
    'effective_hand_strength': True,       # EHS calculations
    'bluff_detection': True,               # Bluff analysis
    'fold_equity_calculation': True        # Advanced fold equity
}
```

#### Blackjack Configuration

```python
# Advanced blackjack settings
blackjack_config = {
    'counting_system': 'hi_lo',            # hi_lo, zen_count, wong_halves
    'true_count_precision': 0.1,           # True count accuracy
    'strategy_deviations': True,           # Count-based deviations
    'betting_strategy': 'kelly',           # kelly, conservative, aggressive
    'risk_management': True,               # Bankroll risk assessment
    'insurance_analysis': True             # Insurance recommendations
}
```

### Performance Tuning

#### Adaptive Performance Settings

```python
performance_tuning = {
    'auto_adjust_simulations': True,       # Reduce sims when slow
    'cache_aggressive': True,              # Cache more results
    'frame_skip_intelligent': True,        # Skip frames when needed
    'memory_optimization': True,           # Clean memory periodically
    'gpu_acceleration': False              # Use GPU if available
}
```

#### Manual Performance Adjustment

If experiencing performance issues:

1. **Reduce simulation count**
   ```python
   # In config/config.py
   performance_config['max_simulation_count'] = 5000  # Default: 10000
   ```

2. **Lower capture rate**
   ```python
   screen_config['capture_rate'] = 15  # Default: 20
   ```

3. **Enable frame skipping**
   ```python
   screen_config['adaptive_delay'] = True
   ```

4. **Increase cache size**
   ```python
   performance_config['cache_enabled'] = True
   ```

### Custom Game Rules

#### Blackjack Rule Variations

```python
blackjack_rules = {
    'decks': 6,                          # Number of decks
    'dealer_hits_soft_17': True,         # H17 (true) or S17 (false)
    'double_after_split': True,          # DAS allowed
    'surrender_allowed': True,           # Late surrender
    'blackjack_payout': 1.5,             # Blackjack payout ratio
    'dealer_peek': True,                 # Dealer checks for blackjack
    'penetration_warning': 0.75          # Deck penetration threshold
}
```

#### Poker Game Variations

```python
poker_rules = {
    'game_type': 'texas_holdem',         # Game variant
    'stakes': 'NLHE',                    # No Limit Hold'em
    'small_blind': 1.0,                  # Small blind amount
    'big_blind': 2.0,                    # Big blind amount
    'ante': 0.0,                         # Ante amount (if any)
    'max_buyin': 100.0,                  # Maximum buy-in
    'player_count': 6                     # Number of players
}
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Cards Not Detected

**Symptoms**: Cards show as "unknown" or low confidence
**Solutions**:
1. Re-run calibration with more precise clicking
2. Ensure good screen lighting
3. Check that cards are clearly visible in game
4. Try adjusting screen brightness/contrast

**Advanced troubleshooting**:
```bash
# Enable debug mode to save screenshots
python main.py --debug

# Check detection confidence thresholds
# In config/config.py, lower thresholds temporarily
game_config['card_detection_threshold'] = 0.6  # Default: 0.8
```

#### Issue 2: Text Recognition Problems

**Symptoms**: Pot sizes, bets show as "???" or incorrect values
**Solutions**:
1. Ensure game text is clear and readable
2. Check that game window isn't too small
3. Try different text colors in game settings

**OCR-specific fixes**:
```python
# Adjust OCR confidence threshold
game_config['ocr_confidence'] = 0.5  # Default: 0.7

# Enable OCR preprocessing (usually automatic)
# Check that EasyOCR is properly installed
python -c "import easyocr; print('EasyOCR OK')"
```

#### Issue 3: Performance Issues

**Symptoms**: Lag, high CPU usage, slow updates
**Solutions**:
1. Reduce simulation counts
2. Lower capture rate
3. Enable adaptive performance
4. Close unnecessary background applications

**Performance commands**:
```bash
# Check system resource usage
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"

# Monitor application performance
# The app will show FPS and performance metrics in debug mode
```

#### Issue 4: Overlay Not Visible

**Symptoms**: Analysis not showing on screen
**Solutions**:
1. Check if agent started successfully
2. Verify overlay position is on visible screen
3. Try different transparency settings
4. Restart application

**Overlay troubleshooting**:
```bash
# Start with visible overlay position
python main.py --overlay-x 10 --overlay-y 10 --transparency 0.5

# Check if overlay process is running
# Look for Python processes in Task Manager/Activity Monitor
```

#### Issue 5: Game Not Detected

**Symptoms**: "No game detected" message persists
**Solutions**:
1. Ensure game window is visible and active
2. Re-calibrate game regions
3. Check that game is supported
4. Try different game window sizes

### Debug Mode Usage

1. **Enable debug logging**
   ```bash
   python main.py --debug
   ```

2. **Check log files**
   ```bash
   # View application logs
   tail -f poker_ai_agent.log
   ```

3. **Save debug screenshots**
   - Screenshots saved as `debug_TIMESTAMP.png`
   - Check these to verify what the AI "sees"

4. **Diagnostic information**
   ```python
   # Get comprehensive diagnostics
   from src.ai_agent import AIAgentManager
   manager = AIAgentManager()
   manager.agent.get_diagnostic_info()
   ```

### Error Messages and Solutions

| Error Message | Likely Cause | Solution |
|---------------|-------------|----------|
| "Screen capture failed" | Permission issues, multiple monitors | Check monitor settings, restart app |
| "OCR initialization failed" | Missing language packs | Reinstall EasyOCR with: `pip install easyocr` |
| "CUDA not available" | GPU acceleration issue | Use CPU mode (default), update GPU drivers |
| "No game regions calibrated" | First run without calibration | Run `python main.py --calibrate` |
| "Memory usage high" | Too many cached results | Restart app, reduce cache settings |

## Performance Optimization

### System Optimization Tips

1. **Graphics Settings**
   - Use native screen resolution
   - Disable unnecessary visual effects
   - Ensure stable frame rate in games

2. **Application Settings**
   - Close unused background applications
   - Disable Windows/macOS animations
   - Use SSD storage for better I/O performance

3. **Network Optimization**
   - Use stable internet connection
   - Disable bandwidth-heavy applications
   - Consider wired connection over WiFi

### Application-Specific Optimizations

#### Memory Usage Optimization

```python
# Reduce memory footprint in config
performance_config = {
    'cache_size_limit': 100,           # Reduce cache size
    'history_retention_days': 7,       # Keep less history
    'max_simulation_count': 5000,      # Reduce simulations
    'cleanup_interval_minutes': 5      # More frequent cleanup
}
```

#### CPU Usage Optimization

```python
# Reduce CPU usage in config
screen_config = {
    'capture_rate': 15,                # Lower FPS
    'adaptive_delay': True,            # Smart delays
    'processing_threads': 2            # Limit threads
}
```

#### GPU Acceleration (Advanced)

```python
# Enable GPU acceleration if available
gpu_config = {
    'enable_cuda': True,               # Use NVIDIA GPU
    'gpu_memory_fraction': 0.3,        # GPU memory usage limit
    'mixed_precision': True            # Faster but less precise
}
```

## Maintenance

### Regular Maintenance Tasks

1. **Weekly**
   - Clear application cache
   - Review and archive old logs
   - Update system dependencies
   - Test calibration accuracy

2. **Monthly**
   - Full system scan for issues
   - Update graphics drivers
   - Review performance metrics
   - Backup configuration files

3. **As Needed**
   - Re-calibrate when changing displays
   - Update when game clients update
   - Clean install if issues persist

### Log Management

```bash
# View recent logs
tail -f poker_ai_agent.log

# Archive old logs
mkdir -p logs/archive
mv poker_ai_agent.log.* logs/archive/

# Clean very old logs (older than 30 days)
find logs/archive -name "*.log.*" -mtime +30 -delete
```

### Backup and Recovery

#### What to Backup

- `config/config.py` (your customizations)
- Calibration data (stored in config)
- Custom model files (if any)
- `.env` file (API keys, settings)

#### Backup Command

```bash
# Create backup archive
tar -czf poker_ai_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
    config/ \
    poker_ai_agent.log \
    .env
```

#### Recovery Process

1. Restore configuration files
2. Reinstall dependencies if needed
3. Re-calibrate game regions
4. Test functionality

### Update Process

1. **Check for updates**
   ```bash
   git fetch origin
   git status
   ```

2. **Backup current configuration**
   ```bash
   cp config/config.py config/config.py.backup
   ```

3. **Apply updates**
   ```bash
   git pull origin main
   ```

4. **Update dependencies**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

5. **Test functionality**
   ```bash
   python main.py --calibrate  # If regions changed
   python main.py               # Test normal operation
   ```

## Support and Help

### Getting Help

1. **Check documentation first**
   - Read this instructions manual
   - Review troubleshooting section
   - Check FAQ and known issues

2. **Debug mode analysis**
   - Run with `--debug` flag
   - Save and analyze screenshots
   - Check application logs

3. **Community support**
   - Check GitHub issues
   - Review similar problems
   - Submit detailed bug reports

### Reporting Issues

When reporting issues, please include:

- **System information**: OS, Python version, hardware specs
- **Application version**: Git commit or version number
- **Error messages**: Full error text and context
- **Steps to reproduce**: Exact sequence that causes issue
- **Screenshots**: Application state, error messages
- **Log files**: Recent `poker_ai_agent.log` entries

### Feature Requests

For feature requests, please provide:

- **Clear description**: What you want to accomplish
- **Use case**: Why this feature would be helpful
- **Examples**: Specific scenarios where it would be used
- **Priority**: How important is this feature

## Quick Reference

### Essential Commands

```bash
# Basic usage
python main.py

# First-time setup
python main.py --calibrate

# Debug mode
python main.py --debug

# Custom overlay
python main.py --overlay-x 100 --overlay-y 50 --transparency 0.8

# Performance check
python -c "from src.ai_agent import AIAgentManager; m = AIAgentManager(); print(m.agent.get_status())"
```

### Important Files

- `main.py` - Application entry point
- `config/config.py` - Main configuration
- `requirements.txt` - Dependencies list
- `poker_ai_agent.log` - Application logs
- `README.md` - Project overview

### Keyboard Shortcuts

- **Ctrl+Alt+T**: Toggle agent on/off
- **Ctrl+Alt+P**: Pause/resume detection
- **Ctrl+C**: Graceful shutdown

This concludes the comprehensive instructions manual. For additional help, please refer to the project documentation or community resources.