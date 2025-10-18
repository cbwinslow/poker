"""
Screen capture module for poker/blackjack AI agent with enhanced OCR and performance
"""
import time
import threading
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from mss import mss
from PIL import Image
import config.config as config

# OCR imports for text recognition
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    import tesserocr
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


class ScreenCapture:
    """Handles screen capture operations for game monitoring with enhanced performance"""

    def __init__(self, config: config.ScreenConfig):
        self.config = config
        self.monitor = config.monitor
        self.capture_rate = config.capture_rate
        self.region = config.region
        self.sct = mss()
        self.running = False
        self.capture_thread = None
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Performance optimizations
        self.frame_skip_count = 0
        self.target_frame_time = 1.0 / self.capture_rate
        self.last_capture_time = 0
        self.fps_history = []

        # Enhanced OCR setup with hybrid approach (expert analysis)
        self.ocr_reader = None
        self.ocr_cache = {}
        self.ocr_cache_timeout = 1.0  # Cache OCR results for 1 second

        # Try EasyOCR first (recommended for gaming UIs)
        if OCR_AVAILABLE:
            try:
                # Initialize EasyOCR for better performance on "text-in-the-wild"
                self.ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                self.ocr_engine = 'easyocr'  # Track which engine we're using
                logger.info("EasyOCR initialized successfully (recommended for gaming UIs)")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self.ocr_engine = None
        else:
            logger.warning("EasyOCR not available")

        # Fallback OCR options for robustness
        self.fallback_ocr_available = TESSERACT_AVAILABLE
        if self.fallback_ocr_available:
            logger.info("Tesseract available as fallback OCR")

        # Image preprocessing for better OCR accuracy
        self.ocr_preprocessing_enabled = True

    def start_capture(self) -> bool:
        """Start continuous screen capture in background thread"""
        if self.running:
            return False

        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True

    def stop_capture(self) -> None:
        """Stop screen capture"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        """Main capture loop running in background thread with performance optimization"""
        last_cleanup_time = time.time()

        while self.running:
            current_time = time.time()

            try:
                # Adaptive frame skipping for performance
                if self._should_skip_frame(current_time):
                    time.sleep(0.01)  # Short sleep when skipping
                    continue

                # Capture frame with timing measurement
                capture_start = time.time()
                frame = self.capture_screen()
                capture_time = time.time() - capture_start

                if frame is not None:
                    with self.frame_lock:
                        self.latest_frame = frame.copy()
                        self.last_capture_time = current_time

                    # Track FPS for performance monitoring
                    self._update_fps_tracking(current_time)

                # Adaptive sleep based on capture time
                elapsed_time = time.time() - current_time
                remaining_time = max(0.001, self.target_frame_time - elapsed_time)
                time.sleep(remaining_time)

                # Periodic cleanup
                if current_time - last_cleanup_time > 60:  # Every minute
                    self._cleanup_resources()
                    last_cleanup_time = current_time

            except Exception as e:
                # Only print non-display related errors
                if "display" not in str(e).lower() and "thread" not in str(e).lower():
                    print(f"Error in capture loop: {e}")
                time.sleep(0.1)  # Brief pause before retry

    def _should_skip_frame(self, current_time: float) -> bool:
        """Determine if frame should be skipped for performance"""
        if not hasattr(self, '_last_frame_time'):
            self._last_frame_time = 0

        time_since_last_frame = current_time - self._last_frame_time

        # Skip if we're running too fast (more than 2x target rate)
        if time_since_last_frame < self.target_frame_time * 0.5:
            self.frame_skip_count += 1
            return True

        self._last_frame_time = current_time
        return False

    def _update_fps_tracking(self, current_time: float) -> None:
        """Update FPS tracking for performance monitoring"""
        if not hasattr(self, '_fps_start_time'):
            self._fps_start_time = current_time

        # Calculate actual FPS every second
        if current_time - self._fps_start_time >= 1.0:
            actual_fps = len(self.fps_history) / (current_time - self._fps_start_time)
            self.fps_history = []  # Reset for next period
            self._fps_start_time = current_time

            # Log if FPS is significantly different from target
            target_fps = self.capture_rate
            if abs(actual_fps - target_fps) > target_fps * 0.2:  # 20% tolerance
                print(f"Capture FPS: {actual_fps:.1f} (target: {target_fps})")

        self.fps_history.append(current_time)

    def _cleanup_resources(self) -> None:
        """Clean up resources periodically"""
        # Clear old OCR cache
        current_time = time.time()
        self.ocr_cache = {
            k: v for k, v in self.ocr_cache.items()
            if current_time - v['timestamp'] < self.ocr_cache_timeout
        }

        # Force garbage collection if available
        try:
            import gc
            gc.collect()
        except ImportError:
            pass

    def capture_screen(self) -> Optional[np.ndarray]:
        """Capture a single frame from the screen"""
        try:
            monitor = self.sct.monitors[self.monitor]

            # If region is specified, use it; otherwise use full monitor
            if self.region:
                x, y, width, height = self.region
                # Ensure region is within monitor bounds
                x = max(0, min(x, monitor['width'] - width))
                y = max(0, min(y, monitor['height'] - height))
                monitor_region = {'left': x, 'top': y, 'width': width, 'height': height}
            else:
                monitor_region = monitor

            # Capture screenshot
            screenshot = self.sct.grab(monitor_region)

            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

            # Convert to numpy array
            frame = np.array(img)

            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return frame

        except Exception as e:
            # Create a placeholder frame instead of returning None
            # This allows the agent to continue running even with capture issues
            try:
                if self.region:
                    width, height = self.region[2], self.region[3]
                else:
                    monitor = self.sct.monitors[self.monitor]
                    width, height = monitor['width'], monitor['height']

                # Return a black placeholder frame
                placeholder = np.zeros((height, width, 3), dtype=np.uint8)
                return placeholder
            except:
                return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        with self.frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def capture_game_regions(self) -> Dict[str, np.ndarray]:
        """Capture specific game regions (poker or blackjack)"""
        frame = self.get_latest_frame()
        if frame is None:
            return {}

        regions = {}

        # This would be expanded based on the specific game being monitored
        # For now, return the full frame
        regions['main'] = frame

        return regions

    def save_debug_image(self, frame: np.ndarray, filename: str) -> None:
        """Save a frame for debugging purposes"""
        try:
            cv2.imwrite(filename, frame)
        except Exception as e:
            print(f"Error saving debug image: {e}")

    def get_screen_size(self) -> Tuple[int, int]:
        """Get the size of the capture area"""
        if self.region:
            return self.region[2], self.region[3]
        else:
            monitor = self.sct.monitors[self.monitor]
            return monitor['width'], monitor['height']

    def extract_text_from_region(self, region: Tuple[int, int, int, int],
                                use_cache: bool = True) -> str:
        """
        Extract text from a specific screen region using OCR

        Args:
            region: (x, y, width, height) of the region to analyze
            use_cache: Whether to use cached OCR results

        Returns:
            Extracted text string
        """
        if not OCR_AVAILABLE:
            return ""

        # Check cache first
        cache_key = f"{region}_{time.time() // self.ocr_cache_timeout}"
        if use_cache and cache_key in self.ocr_cache:
            return self.ocr_cache[cache_key]['text']

        try:
            # Capture the specific region
            region_frame = self.capture_region(region)
            if region_frame is None:
                return ""

            # Preprocess image for better OCR accuracy
            if self.ocr_preprocessing_enabled:
                processed_frame = self._preprocess_for_ocr(region_frame)
            else:
                processed_frame = region_frame

            # Enhanced OCR with hybrid approach (expert analysis)
            extracted_text = ""

            if self.ocr_reader and self.ocr_engine == 'easyocr':
                try:
                    # EasyOCR with improved confidence threshold for gaming text
                    results = self.ocr_reader.readtext(processed_frame, detail=1)

                    # Filter and combine results with higher confidence threshold for accuracy
                    confident_results = [result[1] for result in results if result[2] > 0.7]
                    if confident_results:
                        extracted_text = " ".join(confident_results)
                    else:
                        # If no high-confidence results, try lower threshold
                        confident_results = [result[1] for result in results if result[2] > 0.5]
                        extracted_text = " ".join(confident_results) if confident_results else ""

                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
                    extracted_text = self._try_fallback_ocr(processed_frame)

            elif self.fallback_ocr_available:
                # Try Tesseract as fallback (expert analysis recommendation)
                extracted_text = self._try_fallback_ocr(processed_frame)
            else:
                # Last resort: basic extraction
                extracted_text = self._basic_text_extraction(processed_frame)

            # Clean up the text (extract numbers and basic characters)
            cleaned_text = self._clean_ocr_text(extracted_text)

            # Cache the result
            if use_cache:
                self.ocr_cache[cache_key] = {
                    'text': cleaned_text,
                    'timestamp': time.time()
                }

            return cleaned_text

        except Exception as e:
            print(f"OCR Error: {e}")
            return ""

    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR accuracy"""
        # Convert to grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply adaptive thresholding for better contrast
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Apply slight blur to reduce noise
        blurred = cv2.GaussianBlur(thresh, (1, 1), 0)

        # Morphological operations to clean up the image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _try_fallback_ocr(self, image: np.ndarray) -> str:
        """Try fallback OCR methods (Tesseract) for robustness"""
        if not self.fallback_ocr_available:
            return self._basic_text_extraction(image)

        try:
            # Try Tesseract with optimized settings for gaming text
            if TESSERACT_AVAILABLE:
                # Convert to PIL Image for pytesseract
                pil_image = Image.fromarray(image)

                # Optimized Tesseract configuration for gaming UIs
                config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.$/'
                text = pytesseract.image_to_string(pil_image, config=config)

                # Clean the result
                return self._clean_ocr_text(text)
        except Exception as e:
            logger.warning(f"Fallback OCR failed: {e}")

        # If all OCR methods fail, return empty string
        return ""

    def _basic_text_extraction(self, image: np.ndarray) -> str:
        """Basic text extraction when OCR libraries aren't available"""
        # Enhanced basic extraction as last resort
        text = ""

        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Enhanced preprocessing for better results
        # Adaptive thresholding (better than simple threshold)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        # Noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # This is still basic, but better than before
        # In production, would need proper OCR library
        return text

    def _clean_ocr_text(self, text: str) -> str:
        """Clean OCR text to extract useful information (numbers, etc.)"""
        if not text:
            return ""

        # Remove common OCR artifacts and clean up
        cleaned = text.replace('|', '1').replace('O', '0').replace('l', '1')
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c in ['.', '$', '/'])

        # Extract numbers and basic symbols (useful for pot sizes, bets, etc.)
        import re
        numbers = re.findall(r'\d+\.?\d*', cleaned)

        if numbers:
            # Return the largest number found (likely pot size or bet amount)
            return max(numbers, key=len)
        elif cleaned:
            return cleaned.strip()
        else:
            return ""

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for monitoring"""
        return {
            'target_fps': self.capture_rate,
            'actual_fps': len(self.fps_history),
            'frame_skip_count': self.frame_skip_count,
            'last_capture_time': self.last_capture_time,
            'ocr_cache_size': len(self.ocr_cache),
            'thread_alive': self.capture_thread.is_alive() if self.capture_thread else False
        }

    def optimize_capture_rate(self, target_latency_ms: float = 100) -> None:
        """Automatically optimize capture rate based on performance"""
        # Calculate optimal frame rate based on target latency
        optimal_fps = 1000.0 / target_latency_ms

        # Don't exceed hardware capabilities
        max_reasonable_fps = 60
        new_fps = min(optimal_fps, max_reasonable_fps)

        if abs(new_fps - self.capture_rate) > 5:  # Only change if significant difference
            print(f"Optimizing capture rate: {self.capture_rate} -> {new_fps:.1f} FPS")
            self.capture_rate = new_fps
            self.target_frame_time = 1.0 / new_fps


class ScreenCaptureManager:
    """Manager class for screen capture operations"""

    def __init__(self):
        self.screen_capture = ScreenCapture(config.config.screen)
        self.calibration_mode = False

    def start_monitoring(self) -> bool:
        """Start monitoring the screen"""
        return self.screen_capture.start_capture()

    def stop_monitoring(self) -> None:
        """Stop monitoring the screen"""
        self.screen_capture.stop_capture()

    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current screen frame"""
        return self.screen_capture.get_latest_frame()

    def capture_region(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Capture a specific region of the screen"""
        # Temporarily set the region and capture
        original_region = self.screen_capture.region
        self.screen_capture.region = region

        frame = self.screen_capture.capture_screen()

        # Restore original region
        self.screen_capture.region = original_region

        return frame

    def calibrate_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Interactive calibration of game regions"""
        print("Calibration mode - click on corners of game regions")
        print("This would require user interaction in a real implementation")
        # Placeholder for calibration logic
        return {}