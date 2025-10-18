# ❓ Frequently Asked Questions

## Common Questions and Troubleshooting

This FAQ addresses the most common questions, issues, and concerns about the AI Blackjack Poker Assistant.

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Configuration Issues](#configuration-issues)
3. [Game Detection Problems](#game-detection-problems)
4. [Performance Issues](#performance-issues)
5. [Accuracy and Strategy](#accuracy-and-strategy)
6. [Technical Issues](#technical-issues)
7. [Legal and Ethical Questions](#legal-and-ethical-questions)

## Installation and Setup

### Q: I'm getting import errors when trying to install dependencies. What should I do?

**A:** This is usually due to missing system dependencies or Python version issues. Try these solutions:

1. **Update pip and setuptools:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```

2. **Install system dependencies (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev build-essential
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   ```

3. **Install system dependencies (macOS):**
   ```bash
   brew install opencv
   brew install tesseract
   ```

4. **Install packages one by one:**
   ```bash
   pip install numpy opencv-python
   pip install mss pyautogui
   pip install easyocr torch
   ```

### Q: The installation seems to work, but I get "module not found" errors when running the application.

**A:** This often indicates a Python path or virtual environment issue:

1. **Check your Python environment:**
   ```bash
   python -c "import sys; print(sys.path)"
   python -c "import cv2; print('OpenCV OK')"
   ```

2. **Ensure you're using the correct Python version:**
   ```bash
   python --version  # Should be 3.8+
   pip --version    # Should show the virtual environment
   ```

3. **Try reinstalling in a fresh virtual environment:**
   ```bash
   python -m venv venv_clean
   source venv_clean/bin/activate  # or venv_clean\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

### Q: I'm on Windows and getting errors about missing DLL files or Visual C++ redistributables.

**A:** Windows requires Visual C++ redistributables for some packages:

1. **Download and install:**
   - Visual Studio 2019 C++ redistributables (x64)
   - Visual Studio 2022 C++ redistributables (x64)

2. **Alternative solution:**
   ```bash
   # Install pre-compiled wheels
   pip install --only-binary=all opencv-python
   pip install --only-binary=all easyocr
   ```

## Configuration Issues

### Q: The overlay window doesn't appear or isn't visible.

**A:** Overlay visibility issues are common. Try these solutions:

1. **Check overlay position:**
   ```bash
   python main.py --overlay-x 10 --overlay-y 10 --transparency 0.5
   ```

2. **Verify configuration:**
   ```python
   # Check config/config.py
   ui_config = {
       'overlay_alpha': 0.9,  # Try 0.5 for testing
       'overlay_position': (10, 10),  # Try (0, 0)
   }
   ```

3. **Test on different monitor:**
   ```python
   screen_config = {
       'monitor': 0,  # Try 1 for second monitor
   }
   ```

### Q: The calibration process isn't working properly.

**A:** Calibration issues usually relate to clicking accuracy or game window setup:

1. **Ensure proper game window setup:**
   - Game window should be clearly visible
   - Cards and text should be readable
   - No other windows overlapping the game

2. **Click calibration points carefully:**
   - Click the top-left corner of regions, not the center
   - Click the bottom-right corner, not the center
   - Take your time and be precise

3. **Re-run calibration:**
   ```bash
   python main.py --calibrate
   ```

### Q: My game isn't being detected even after calibration.

**A:** Game detection issues can have several causes:

1. **Check game compatibility:**
   - Ensure your game client is supported
   - Verify game is running and visible
   - Try different game window sizes

2. **Adjust detection thresholds:**
   ```python
   game_config = {
       'card_detection_threshold': 0.6,  # Lower from default 0.8
       'ocr_confidence': 0.5,           # Lower from default 0.7
   }
   ```

3. **Enable debug mode:**
   ```bash
   python main.py --debug
   # Check saved screenshots in the application directory
   ```

## Game Detection Problems

### Q: Cards are showing as "unknown" or with low confidence.

**A:** Card detection issues are usually related to image quality or calibration:

1. **Improve image quality:**
   - Ensure good lighting on your screen
   - Avoid screen glare or reflections
   - Clean your screen if necessary

2. **Re-calibrate card regions:**
   ```bash
   python main.py --calibrate
   # Click more precisely on card corners
   ```

3. **Check game settings:**
   - Increase card size in game settings
   - Use standard card designs
   - Disable card animations

### Q: Text recognition (pot sizes, bets) isn't working.

**A:** OCR issues are common with different text styles and sizes:

1. **Check text clarity:**
   - Ensure bet amounts and pot sizes are clearly visible
   - Try different text colors in game settings
   - Increase font size if possible

2. **Adjust OCR settings:**
   ```python
   game_config = {
       'ocr_confidence': 0.5,  # Lower threshold
   }
   ```

3. **Test OCR separately:**
   ```python
   from src.screen_capture import ScreenCaptureManager
   manager = ScreenCaptureManager()
   text = manager.extract_text_from_region((x, y, width, height))
   print(f"Extracted text: {text}")
   ```

### Q: The AI detects the wrong game or switches between poker and blackjack incorrectly.

**A:** Game type detection issues:

1. **Calibrate for one game at a time:**
   - Run calibration separately for each game
   - Ensure only one game is visible during calibration

2. **Adjust detection confidence:**
   ```python
   # Increase threshold for more certainty
   game_config = {
       'card_detection_threshold': 0.9,
   }
   ```

3. **Manual game type forcing:**
   ```python
   # Temporarily force game type for testing
   agent.current_game_type = 'poker'  # or 'blackjack'
   ```

## Performance Issues

### Q: The application is using too much CPU or running slowly.

**A:** Performance issues can be addressed through configuration:

1. **Reduce capture rate:**
   ```python
   screen_config = {
       'capture_rate': 15,  # Reduce from default 20
   }
   ```

2. **Enable adaptive performance:**
   ```python
   screen_config = {
       'adaptive_delay': True,
   }
   performance_config = {
       'max_simulation_count': 5000,  # Reduce from default 10000
   }
   ```

3. **Enable caching:**
   ```python
   performance_config = {
       'cache_enabled': True,
   }
   ```

### Q: The analysis seems slow or delayed.

**A:** Latency issues:

1. **Check system resources:**
   ```bash
   # Monitor resource usage
   python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"
   ```

2. **Optimize for your system:**
   ```python
   performance_config = {
       'max_simulation_count': 5000,
       'cache_size': 1000,
   }
   ```

3. **Close unnecessary applications:**
   - Close web browsers, streaming apps, etc.
   - Disable background applications

### Q: Memory usage keeps increasing over time.

**A:** Memory leak issues:

1. **Enable garbage collection:**
   ```python
   performance_config = {
       'enable_gc': True,
       'gc_threshold': 0.8,
   }
   ```

2. **Reduce cache sizes:**
   ```python
   performance_config = {
       'cache_size': 100,  # Reduce cache retention
   }
   ```

3. **Restart application periodically:**
   - Plan to restart every few hours during extended play
   - Monitor memory usage and restart if >500MB

## Accuracy and Strategy

### Q: The AI's strategy recommendations don't match what I learned from books.

**A:** Strategy differences can have several explanations:

1. **Different rule sets:**
   - Books often assume specific casino rules
   - Check if your casino uses H17 vs S17, DAS, etc.
   - Verify the AI is configured for your casino's rules

2. **Card counting deviations:**
   - The AI may be using count-based strategy deviations
   - Check the true count display
   - Understand that optimal play changes with count

3. **Advanced strategy:**
   - The AI uses research-based algorithms
   - Some recommendations may be more sophisticated than basic strategy

### Q: The AI says one thing but my "gut feeling" says another. Which should I trust?

**A:** Trust the math, not your gut:

1. **Understand the reasoning:**
   - The AI display shows probabilities and expected values
   - Learn to interpret the mathematical reasoning
   - Use it as a learning tool to improve your understanding

2. **Track results over time:**
   - Keep a log of hands where you followed vs. ignored the AI
   - Compare results after 100+ hands
   - Let the data guide your decisions

3. **Start with small bets:**
   - Use the AI as a learning tool initially
   - Gradually increase confidence as you verify its accuracy

### Q: How do I know if the AI is giving good advice?

**A:** Validate the AI's performance:

1. **Track accuracy:**
   - Compare AI recommendations with known optimal strategy
   - Use strategy cards or trainers to verify
   - Track how often the AI is correct

2. **Long-term results:**
   - Monitor your results when following AI advice
   - Compare with periods when you don't use it
   - Focus on expected value, not individual hands

3. **Calibration verification:**
   ```bash
   python main.py --debug
   # Review saved screenshots and detection results
   ```

## Technical Issues

### Q: I'm getting Python errors I don't understand. What should I do?

**A:** Systematic troubleshooting:

1. **Check the error logs:**
   ```bash
   tail -f poker_ai_agent.log
   ```

2. **Isolate the issue:**
   - Test individual components
   - Try the application with minimal configuration
   - Check if issue occurs with different games

3. **Common error solutions:**
   ```bash
   # Permission errors
   # Run as administrator or check file permissions

   # Import errors
   # Reinstall dependencies or check Python path

   # Memory errors
   # Restart application or reduce cache sizes
   ```

### Q: The application crashes or freezes frequently.

**A:** Stability issues:

1. **Update your system:**
   - Ensure Windows/macOS/Linux is up to date
   - Update graphics drivers
   - Install latest Windows updates

2. **Check system resources:**
   ```bash
   # Monitor system resources
   python -c "import psutil; print(psutil.virtual_memory())"
   ```

3. **Application-specific fixes:**
   ```python
   performance_config = {
       'enable_gc': True,
       'gc_threshold': 0.7,
       'error_recovery': True,
   }
   ```

### Q: I'm having issues with multiple monitors or different screen resolutions.

**A:** Multi-monitor setup issues:

1. **Specify correct monitor:**
   ```python
   screen_config = {
       'monitor': 0,  # 0 = primary, 1 = secondary
   }
   ```

2. **Check resolution compatibility:**
   - Ensure game runs at native resolution
   - Try 1920x1080 if having issues
   - Avoid scaled resolutions (125%, 150%, etc.)

3. **Calibration for specific monitor:**
   ```bash
   python main.py --calibrate
   # Calibrate on the monitor where the game is displayed
   ```

## Legal and Ethical Questions

### Q: Is using this AI assistant considered cheating?

**A:** It depends on the context and platform:

1. **Educational use:** ✅ Perfectly acceptable
   - Learning game theory and strategy
   - Understanding mathematical principles
   - Personal skill development

2. **Play money games:** ✅ Generally acceptable
   - Most platforms allow assistance in play money
   - Good for practice and learning

3. **Real money online:** ⚠️ Check platform rules
   - Some platforms prohibit external assistance
   - Others may allow strategy tools
   - Always check terms of service

4. **Live casino:** ❌ Not recommended
   - Physical casinos prohibit external devices
   - Could be considered advantage play
   - Legal and ethical concerns

### Q: Can online poker sites detect this software?

**A:** Detection risk varies by platform:

1. **Detection methods:**
   - Behavioral analysis (unusual timing patterns)
   - Screen scraping detection
   - Process monitoring

2. **Risk mitigation:**
   - Use only for educational purposes
   - Avoid real-money play if concerned
   - Follow platform terms of service

3. **Safe usage:**
   - Use for play money games only
   - Study strategy away from tables
   - Focus on learning rather than winning

### Q: Is this legal to use?

**A:** Legality depends on jurisdiction and usage:

1. **Educational/research use:** ✅ Generally legal
   - Personal learning and study
   - Academic research
   - Software development

2. **Commercial gambling:** ⚠️ Varies by location
   - Some jurisdictions restrict gambling aids
   - Others have specific laws about poker tools
   - Always check local gambling laws

3. **Platform policies:** ⚠️ Terms of service
   - Online platforms have their own rules
   - Violation can result in account bans
   - Funds may be forfeited

### Q: What are the ethical considerations?

**A:** Important ethical guidelines:

1. **Transparency:** Be honest about using tools
2. **Education:** Use primarily for learning
3. **Responsibility:** Don't exploit recreational players
4. **Legality:** Comply with all applicable laws and regulations

## Advanced Troubleshooting

### Q: How can I debug detection issues?

**A:** Comprehensive debugging process:

1. **Enable debug mode:**
   ```bash
   python main.py --debug
   ```

2. **Review saved images:**
   - Check `debug_*.png` files in application directory
   - Verify what the AI "sees" vs. actual screen

3. **Check detection confidence:**
   ```python
   diagnostics = manager.get_diagnostic_info()
   print(f"Detection confidence: {diagnostics['agent_status']['detection_confidence']}")
   ```

4. **Test individual components:**
   ```python
   from src.screen_capture import ScreenCaptureManager
   manager = ScreenCaptureManager()
   frame = manager.get_current_frame()
   # Analyze frame properties
   ```

### Q: The AI recommendations seem inconsistent. What's wrong?

**A:** Inconsistency issues:

1. **Configuration drift:**
   - Ensure consistent game rules configuration
   - Verify calibration hasn't changed
   - Check for system updates that might affect performance

2. **Detection variability:**
   - Improve calibration precision
   - Ensure stable lighting conditions
   - Check for screen scaling issues

3. **Strategy validation:**
   - Compare with known strategy charts
   - Verify rule configuration matches your game
   - Check if count-based deviations are appropriate

### Q: How do I optimize the AI for my specific setup?

**A:** Customization and optimization:

1. **Profile your system:**
   ```bash
   python -c "import psutil; print(f'CPU cores: {psutil.cpu_count()}'); print(f'RAM: {psutil.virtual_memory().total // 1024 // 1024}MB')"
   ```

2. **Optimize for your hardware:**
   ```python
   if system_has_good_cpu():
       performance_config['max_simulation_count'] = 15000
   else:
       performance_config['max_simulation_count'] = 5000

   if system_has_gpu():
       performance_config['gpu_acceleration'] = True
   ```

3. **Customize for your games:**
   ```python
   # Adjust for your specific game client
   game_config = {
       'card_detection_threshold': 0.85,  # Tune for your game
       'ocr_confidence': 0.7,            # Tune for your text clarity
   }
   ```

## Getting Additional Help

### Community Resources

1. **Documentation:** Refer to the comprehensive documentation suite
2. **GitHub Issues:** Check existing issues and solutions
3. **Discussions:** Participate in community discussions
4. **Tutorials:** Look for video tutorials and guides

### Professional Support

For enterprise or commercial use:
- Contact the development team for custom implementations
- Consider professional training and consultation
- Explore licensing options for commercial deployment

### Contributing

Help improve the project:
- Report bugs with detailed information
- Suggest features with clear use cases
- Contribute code improvements
- Help with documentation and translations

---

**Still need help?** Please provide detailed information about your issue including:
- Operating system and version
- Python version
- Game client and version
- Complete error messages
- Steps to reproduce the issue
- Screenshots if applicable

This FAQ is regularly updated based on user feedback and common issues. Check back regularly for new solutions and troubleshooting tips.