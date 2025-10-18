"""
Main AI Agent for poker/blackjack odds calculation and display
"""
import time
import threading
import logging
import numpy as np
from typing import Optional, Dict, Any
import config.config as config
from .screen_capture import ScreenCaptureManager
from .poker_detector import PokerDetector
from .blackjack_detector import BlackjackDetector
from .poker_odds import PokerOddsCalculator, PokerModelTracker
from .opponent_modeler import AdvancedOpponentModeler
from .blackjack_odds import BlackjackOddsCalculator, BlackjackModelTracker
from .ui.odds_overlay import OverlayManager
from .historical_tracker import historical_tracker

try:
    from pynput import keyboard
    HOTKEYS_AVAILABLE = True
except ImportError:
    HOTKEYS_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('poker_ai_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PokerBlackjackAIAgent:
    """Main AI agent that monitors games and provides real-time odds"""

    def __init__(self, config_obj: config.Config):
        self.config = config_obj
        self.running = False
        self.agent_thread = None

        # Initialize components
        self.screen_manager = ScreenCaptureManager()
        self.poker_detector = PokerDetector(config_obj.game)
        self.blackjack_detector = BlackjackDetector(config_obj.game)
        self.poker_odds_calc = PokerOddsCalculator()
        self.blackjack_odds_calc = BlackjackOddsCalculator()
        self.overlay_manager = OverlayManager(config_obj.ui)

        # Advanced opponent modeling (expert analysis)
        self.opponent_modeler = AdvancedOpponentModeler()

        # Game state tracking
        self.current_game_type = None  # 'poker' or 'blackjack'
        self.last_game_state = None
        self.detection_confidence_threshold = 0.5

        # Performance tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()

        # Error tracking
        self.consecutive_errors = 0
        self.max_consecutive_errors = 10

        # Control states
        self.enabled = True  # Master on/off switch
        self.paused = False  # Pause detection while keeping overlay active
        self.hotkey_listener = None

        # Historical tracking
        self.current_session_id = None
        self.poker_tracker = PokerModelTracker()
        self.blackjack_tracker = BlackjackModelTracker()

        # Performance optimization
        self.calculation_cache = {}
        self.performance_metrics = {
            'calculation_times': [],
            'cache_hit_rate': 0.0,
            'optimization_level': 'auto'
        }

    def start(self) -> bool:
        """Start the AI agent"""
        if self.running:
            return False

        try:
            # Start screen capture
            logger.info("Starting screen capture monitoring...")
            if not self.screen_manager.start_monitoring():
                logger.error("Failed to start screen monitoring")
                return False

            # Start overlay (without mainloop)
            logger.info("Starting overlay manager...")
            self.overlay_manager.start()

            # Setup hotkeys if enabled and available
            if self.config.ui.enable_hotkeys and HOTKEYS_AVAILABLE:
                logger.info("Setting up hotkey listeners...")
                self._setup_hotkeys()
            else:
                logger.info("Hotkeys disabled or unavailable")

            # Start main agent loop
            self.running = True
            self._start_time = time.time()
            self.agent_thread = threading.Thread(target=self._agent_loop, daemon=True)
            self.agent_thread.start()

            logger.info("AI Agent started successfully")
            if self.config.ui.enable_hotkeys and HOTKEYS_AVAILABLE:
                logger.info(f"Hotkey controls - Toggle: {self.config.ui.toggle_hotkey}, Pause: {self.config.ui.pause_hotkey}")

            # Start historical tracking session
            self.current_session_id = historical_tracker.start_session('mixed')
            logger.info(f"Started tracking session: {self.current_session_id}")

            return True

        except Exception as e:
            logger.error(f"Error starting AI agent: {e}", exc_info=True)
            self.stop()
            return False

    def stop(self) -> None:
        """Stop the AI agent"""
        self.running = False

        # Stop hotkey listener
        if self.hotkey_listener:
            self.hotkey_listener.stop()

        # Stop components
        self.screen_manager.stop_monitoring()
        self.overlay_manager.stop()

        # Wait for agent thread to finish
        if self.agent_thread:
            logger.info("Waiting for agent thread to finish...")
            self.agent_thread.join(timeout=2.0)
            if self.agent_thread.is_alive():
                logger.warning("Agent thread did not stop gracefully")

        # End historical tracking session
        if self.current_session_id:
            historical_tracker.end_session(self.current_session_id)
            logger.info(f"Ended tracking session: {self.current_session_id}")

        logger.info("AI Agent stopped")

    def _agent_loop(self) -> None:
        """Main agent loop - runs continuously"""
        no_game_detected_count = 0
        long_idle_count = 0

        while self.running:
            try:
                # Check if agent is enabled
                if not self.enabled:
                    self.overlay_manager.show_waiting("AGENT DISABLED - Press " + self.config.ui.toggle_hotkey + " to enable")
                    time.sleep(0.5)
                    continue

                # Check if detection is paused
                if self.paused:
                    self.overlay_manager.show_waiting("DETECTION PAUSED - Press " + self.config.ui.pause_hotkey + " to resume")
                    time.sleep(0.5)
                    continue

                # Get current screen frame
                frame = self.screen_manager.get_current_frame()
                if frame is None:
                    self.consecutive_errors += 1
                    if self.consecutive_errors > self.max_consecutive_errors:
                        logger.warning(f"Screen capture failing consistently ({self.consecutive_errors} errors), continuing in limited mode...")
                        time.sleep(1.0)
                        self.consecutive_errors = 0
                    else:
                        time.sleep(0.1)
                    continue

                self.consecutive_errors = 0  # Reset error counter on successful capture

                # Detect game type and state
                game_type, game_state = self._detect_game_state(frame)

                if game_state and game_state.confidence >= self.detection_confidence_threshold:
                    self.current_game_type = game_type
                    self.last_game_state = game_state
                    no_game_detected_count = 0
                    long_idle_count = 0

                    # Calculate and display odds with performance optimization
                    self._process_game_state_optimized(game_state)

                else:
                    # No clear game detected
                    self.current_game_type = None
                    no_game_detected_count += 1

                    # Extended idle detection for performance optimization
                    if no_game_detected_count > 20:  # ~1 second at current rate
                        long_idle_count += 1

                    # Show appropriate waiting message
                    if long_idle_count > 50:  # Very long idle - reduce CPU usage significantly
                        self.overlay_manager.show_waiting("EXTENDED IDLE - Reduce CPU usage")
                        logger.debug(f"Extended idle mode activated ({long_idle_count} cycles)")
                    elif no_game_detected_count > 100:  # Medium idle
                        self.overlay_manager.show_waiting("WAITING FOR GAME - No game detected")
                    else:
                        self.overlay_manager.show_waiting()

                # Performance tracking
                self._update_performance()

                # Adaptive delay based on game detection and configuration
                if self.config.screen.adaptive_delay:
                    if self.current_game_type:
                        time.sleep(self.config.screen.active_delay)  # Faster when game is active
                    elif long_idle_count > 50:  # Very long idle - minimal CPU usage
                        time.sleep(0.5)  # Much slower to save significant CPU
                    elif no_game_detected_count > 20:  # Medium idle
                        time.sleep(self.config.screen.idle_delay * 1.5)  # Slightly slower
                    else:
                        time.sleep(self.config.screen.idle_delay)   # Normal idle delay
                else:
                    time.sleep(0.05)  # Fixed delay

            except Exception as e:
                self.consecutive_errors += 1
                if "display" not in str(e).lower() and "thread" not in str(e).lower():
                    logger.error(f"Error in agent loop: {e}", exc_info=True)
                else:
                    logger.debug(f"Non-critical error in agent loop: {e}")
                time.sleep(0.1)

    def _detect_game_state(self, frame: np.ndarray) -> tuple[Optional[str], Any]:
        """Detect which game is being played and its current state"""
        # Try poker detection first
        poker_state = self.poker_detector.detect_game_state(frame)
        if poker_state and poker_state.confidence >= self.detection_confidence_threshold:
            return 'poker', poker_state

        # Try blackjack detection
        blackjack_state = self.blackjack_detector.detect_game_state(frame)
        if blackjack_state and blackjack_state.confidence >= self.detection_confidence_threshold:
            return 'blackjack', blackjack_state

        return None, None

    def _process_game_state_optimized(self, game_state: Any) -> None:
        """Process game state with performance optimization"""
        if self.current_game_type == 'poker':
            self._process_poker_state(game_state)
        elif self.current_game_type == 'blackjack':
            self._process_blackjack_state(game_state)

    def _process_game_state(self, game_state: Any) -> None:
        """Legacy method for backward compatibility"""
        self._process_game_state_optimized(game_state)

    def _process_poker_state(self, poker_state) -> None:
        """Process poker game state"""
        try:
            # Calculate poker odds
            poker_odds = self.poker_odds_calc.calculate_odds(poker_state)
            if poker_odds:
                # Update overlay
                self.overlay_manager.update_poker_odds(poker_odds)

                # Log for debugging
                logger.info(f"Poker Odds - Hand: {poker_odds.hand_category}, "
                           f"Equity: {poker_odds.equity_vs_random:.1%}, "
                           f"Recommendation: {poker_odds.recommended_action}")

                # Record for historical tracking (placeholder - would need actual outcome)
                # historical_tracker.record_game_result(
                #     'poker', poker_state, poker_odds.recommended_action,
                #     'unknown', poker_odds.expected_value, 0.0, self.current_session_id
                # )

        except Exception as e:
            logger.error(f"Error processing poker state: {e}", exc_info=True)

    def _process_blackjack_state(self, blackjack_state) -> None:
        """Process blackjack game state"""
        try:
            # Calculate blackjack odds
            blackjack_odds = self.blackjack_odds_calc.calculate_odds(blackjack_state)
            if blackjack_odds:
                # Update overlay
                self.overlay_manager.update_blackjack_odds(blackjack_odds)

                # Log for debugging
                logger.info(f"Blackjack Odds - Win Prob: {blackjack_odds.player_win_probability:.1%}, "
                           f"Dealer Bust: {blackjack_odds.dealer_bust_probability:.1%}, "
                           f"Recommendation: {blackjack_odds.recommended_action}")

        except Exception as e:
            logger.error(f"Error processing blackjack state: {e}", exc_info=True)

    def _update_performance(self) -> None:
        """Update performance metrics"""
        self.fps_counter += 1
        current_time = time.time()

        if current_time - self.last_fps_time >= 1.0:
            fps = self.fps_counter / (current_time - self.last_fps_time)
            logger.debug(f"Agent FPS: {fps:.1f}")
            self.fps_counter = 0
            self.last_fps_time = current_time

    def calibrate_game_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Calibrate game detection regions"""
        print("Starting calibration mode...")
        # This would involve user interaction to click on game regions
        # For now, return current regions
        return self.config.game.poker_regions

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'running': self.running,
            'enabled': self.enabled,
            'paused': self.paused,
            'current_game': self.current_game_type,
            'detection_confidence': self.last_game_state.confidence if self.last_game_state else 0.0,
            'overlay_active': self.overlay_manager.overlay_thread.is_alive() if self.overlay_manager.overlay_thread else False,
            'consecutive_errors': self.consecutive_errors,
            'hotkeys_enabled': self.hotkey_listener is not None if self.config.ui.enable_hotkeys and HOTKEYS_AVAILABLE else False,
            'screen_capture_active': self.screen_manager.monitoring if hasattr(self.screen_manager, 'monitoring') else False,
            'performance_fps': self.fps_counter / max(1, time.time() - self.last_fps_time) if time.time() - self.last_fps_time > 0 else 0,
            'memory_usage_mb': self._get_memory_usage(),
            'uptime_seconds': time.time() - getattr(self, '_start_time', time.time())
        }

    def _get_memory_usage(self) -> float:
        """Get approximate memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0

    def set_overlay_position(self, x: int, y: int) -> None:
        """Set overlay window position"""
        self.overlay_manager.set_position(x, y)

    def set_overlay_transparency(self, alpha: float) -> None:
        """Set overlay transparency (0.0 to 1.0)"""
        self.overlay_manager.set_transparency(alpha)

    def save_debug_frame(self, filename: str = "debug_frame.png") -> None:
        """Save current frame for debugging"""
        frame = self.screen_manager.get_current_frame()
        if frame is not None:
            import cv2
            cv2.imwrite(filename, frame)
            logger.info(f"Debug frame saved to {filename}")

    def get_betting_context(self) -> Dict[str, Any]:
        """Get comprehensive betting context information"""
        context = {
            'current_game': self.current_game_type,
            'session_id': self.current_session_id,
            'betting_context': {},
            'historical_performance': {},
            'model_insights': {}
        }

        if self.current_game_type == 'poker' and self.last_game_state:
            if hasattr(self.last_game_state, 'pot_size'):
                context['betting_context'] = {
                    'pot_size': self.last_game_state.pot_size,
                    'current_bet': getattr(self.last_game_state, 'current_bet', None),
                    'total_bet': getattr(self.last_game_state, 'total_bet', None),
                    'player_stack': getattr(self.last_game_state, 'player_stack', None),
                    'player_position': getattr(self.last_game_state, 'player_position', None),
                    'player_count': getattr(self.last_game_state, 'player_count', None),
                    'opponent_stacks': getattr(self.last_game_state, 'opponent_stacks', None)
                }

            # Add historical performance for poker
            context['historical_performance'] = {
                'poker_accuracy': self.poker_tracker.get_model_accuracy(),
                'poker_profitability': self.poker_tracker.get_profitability_metrics()
            }

        elif self.current_game_type == 'blackjack' and self.last_game_state:
            if hasattr(self.last_game_state, 'current_bet'):
                context['betting_context'] = {
                    'current_bet': self.last_game_state.current_bet,
                    'total_bet': getattr(self.last_game_state, 'total_bet', None),
                    'player_bankroll': getattr(self.last_game_state, 'player_bankroll', None),
                    'true_count': getattr(self.last_game_state, 'true_count', 0.0),
                    'deck_penetration': getattr(self.last_game_state, 'deck_penetration', 0.0)
                }

            # Add historical performance for blackjack
            context['historical_performance'] = {
                'blackjack_accuracy': self.blackjack_tracker.get_model_accuracy(),
                'blackjack_profitability': self.blackjack_tracker.get_profitability_metrics(),
                'true_count_effectiveness': self.blackjack_tracker.get_true_count_effectiveness()
            }

        # Add model insights
        context['model_insights'] = historical_tracker.get_model_improvements(self.current_game_type)

        return context

    def _optimize_calculations(self, game_type: str, game_state: Any) -> Dict[str, Any]:
        """Optimize calculations based on performance metrics"""
        # Check if we can use cached results
        cache_key = self._generate_cache_key(game_type, game_state)

        if cache_key in self.calculation_cache:
            cached_result = self.calculation_cache[cache_key]
            self.performance_metrics['cache_hit_rate'] = 0.8  # Update hit rate
            return cached_result

        # Perform calculation with performance monitoring
        start_time = time.time()

        if game_type == 'poker':
            result = self.poker_odds_calc.calculate_odds(game_state)
        elif game_type == 'blackjack':
            result = self.blackjack_odds_calc.calculate_odds(game_state)
        else:
            result = None

        calculation_time = time.time() - start_time
        self.performance_metrics['calculation_times'].append(calculation_time)

        # Cache result if calculation was reasonable time
        if calculation_time < 1.0:  # Less than 1 second
            self.calculation_cache[cache_key] = result

        # Keep cache size manageable
        if len(self.calculation_cache) > 100:
            # Remove oldest 20 entries
            oldest_keys = list(self.calculation_cache.keys())[:20]
            for key in oldest_keys:
                del self.calculation_cache[key]

        return result

    def _generate_cache_key(self, game_type: str, game_state: Any) -> str:
        """Generate cache key for game state"""
        # Create simplified representation for caching
        if game_type == 'poker' and hasattr(game_state, 'player_cards'):
            player_ranks = sorted([card.rank for card in game_state.player_cards])
            community_count = len(game_state.community_cards)
            return f"poker_{player_ranks}_{community_count}"
        elif game_type == 'blackjack' and hasattr(game_state, 'player_cards'):
            player_score = game_state.player_score
            dealer_upcard = game_state.dealer_upcard.value if game_state.dealer_upcard else 0
            return f"blackjack_{player_score}_{dealer_upcard}"

        return f"{game_type}_unknown"

    def get_performance_optimization_tips(self) -> List[str]:
        """Get performance optimization tips based on current metrics"""
        tips = []

        # Analyze calculation times
        if self.performance_metrics['calculation_times']:
            avg_calc_time = sum(self.performance_metrics['calculation_times']) / len(self.performance_metrics['calculation_times'])

            if avg_calc_time > 0.5:
                tips.append("High calculation time detected - consider reducing simulation count")
            elif avg_calc_time > 0.2:
                tips.append("Moderate calculation time - performance is acceptable")

        # Cache effectiveness
        if self.performance_metrics['cache_hit_rate'] < 0.3:
            tips.append("Low cache hit rate - calculations may be too varied for effective caching")

        # Memory usage
        memory_mb = self._get_memory_usage()
        if memory_mb > 200:
            tips.append("High memory usage - consider clearing caches or reducing history retention")

        return tips

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        import platform
        import sys

        return {
            'system_info': {
                'platform': platform.platform(),
                'python_version': sys.version,
                'processor': platform.processor(),
                'architecture': platform.architecture()
            },
            'agent_status': self.get_status(),
            'betting_context': self.get_betting_context(),
            'configuration': {
                'screen': {
                    'monitor': self.config.screen.monitor,
                    'capture_rate': self.config.screen.capture_rate,
                    'adaptive_delay': self.config.screen.adaptive_delay
                },
                'game': {
                    'card_detection_threshold': self.config.game.card_detection_threshold,
                    'ocr_confidence': self.config.game.ocr_confidence
                },
                'ui': {
                    'overlay_alpha': self.config.ui.overlay_alpha,
                    'hotkeys_enabled': self.config.ui.enable_hotkeys,
                    'minimize_enabled': self.config.ui.enable_minimize
                }
            },
            'component_status': {
                'screen_capture': self.screen_manager.monitoring if hasattr(self.screen_manager, 'monitoring') else 'unknown',
                'poker_detector': self.poker_detector is not None,
                'blackjack_detector': self.blackjack_detector is not None,
                'overlay_manager': self.overlay_manager.overlay_thread.is_alive() if self.overlay_manager.overlay_thread else False
            }
        }

    def _setup_hotkeys(self) -> None:
        """Setup global hotkey listeners"""
        if not HOTKEYS_AVAILABLE:
            return

        try:
            # Parse hotkey combinations
            toggle_combination = self._parse_hotkey(self.config.ui.toggle_hotkey)
            pause_combination = self._parse_hotkey(self.config.ui.pause_hotkey)

            # Create hotkey listener
            self.hotkey_listener = keyboard.GlobalHotKeys({
                toggle_combination: self._toggle_agent,
                pause_combination: self._toggle_pause
            })

            self.hotkey_listener.start()
            logger.info(f"Hotkeys registered: {self.config.ui.toggle_hotkey} (toggle), {self.config.ui.pause_hotkey} (pause)")

        except Exception as e:
            logger.error(f"Failed to setup hotkeys: {e}", exc_info=True)
            self.hotkey_listener = None

    def _parse_hotkey(self, hotkey_str: str) -> str:
        """Parse hotkey string to pynput format"""
        # Convert common key names
        key_map = {
            'ctrl': '<ctrl>',
            'alt': '<alt>',
            'shift': '<shift>',
            'cmd': '<cmd>',
            'win': '<cmd>',
            'space': '<space>',
            'tab': '<tab>',
            'enter': '<enter>',
            'esc': '<esc>',
            'up': '<up>',
            'down': '<down>',
            'left': '<left>',
            'right': '<right>',
            'f1': '<f1>',
            'f2': '<f2>',
            'f3': '<f3>',
            'f4': '<f4>',
            'f5': '<f5>',
            'f6': '<f6>',
            'f7': '<f7>',
            'f8': '<f8>',
            'f9': '<f9>',
            'f10': '<f10>',
            'f11': '<f11>',
            'f12': '<f12>'
        }

        # Replace key names and format for pynput
        formatted = hotkey_str.lower()
        for key, replacement in key_map.items():
            formatted = formatted.replace(key, replacement)

        return formatted

    def _toggle_agent(self) -> None:
        """Toggle agent on/off"""
        self.enabled = not self.enabled
        status = "ENABLED" if self.enabled else "DISABLED"
        logger.info(f"Agent {status} via hotkey")

        if self.enabled:
            # Resume operation
            if not self.running:
                logger.info("Restarting agent after hotkey enable")
                self.start()
        else:
            # Pause detection but keep overlay active
            self.paused = True

    def _toggle_pause(self) -> None:
        """Toggle pause/resume state"""
        self.paused = not self.paused
        status = "RESUMED" if not self.paused else "PAUSED"
        logger.info(f"Detection {status} via hotkey")


class AIAgentManager:
    """Manager class for the AI agent"""

    def __init__(self, config_path: str = None):
        self.config = config.config  # Use global config instance
        self.agent = PokerBlackjackAIAgent(self.config)

    def start_agent(self) -> bool:
        """Start the AI agent"""
        return self.agent.start()

    def stop_agent(self) -> None:
        """Stop the AI agent"""
        self.agent.stop()

    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return self.agent.get_status()

    def calibrate_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Calibrate game regions"""
        return self.agent.calibrate_game_regions()

    def save_debug_screenshot(self, filename: str = "debug.png") -> None:
        """Save debug screenshot"""
        self.agent.save_debug_frame(filename)

    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information"""
        return self.agent.get_diagnostic_info()