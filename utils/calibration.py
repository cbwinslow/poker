#!/usr/bin/env python3
"""
Interactive calibration utility for poker/blackjack AI agent
Helps users define accurate game regions for better detection
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
import time
from typing import Dict, Tuple, List, Optional
import config.config as config


class RegionCalibrator:
    """Interactive calibration tool for defining game regions"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()  # Hide the main window
        self.calibration_points = []
        self.current_region = None
        self.regions = {}

    def calibrate_poker_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Calibrate poker game regions interactively"""
        print("Poker Region Calibration")
        print("=" * 50)

        regions_to_calibrate = {
            'player_cards': "Player's hole cards area",
            'community_cards': "Community cards (flop, turn, river)",
            'pot_info': "Pot size and betting information",
            'player_info': "Player stack and position info"
        }

        self.regions = {}

        for region_name, description in regions_to_calibrate.items():
            print(f"\nCalibrating: {region_name}")
            print(f"Description: {description}")
            print("Click and drag to select the region on your screen")
            print("Press Enter when ready...")

            input()  # Wait for user to be ready

            region = self._capture_region_interactive()
            if region:
                self.regions[region_name] = region
                print(f"✓ Calibrated {region_name}: {region}")
            else:
                print(f"✗ Failed to calibrate {region_name}")
                # Use default region
                self.regions[region_name] = self._get_default_poker_region(region_name)

        return self.regions

    def calibrate_blackjack_regions(self) -> Dict[str, Tuple[int, int, int, int]]:
        """Calibrate blackjack game regions interactively"""
        print("\nBlackjack Region Calibration")
        print("=" * 50)

        regions_to_calibrate = {
            'player_cards': "Player's cards area",
            'dealer_cards': "Dealer's cards area",
            'dealer_upcard': "Dealer's visible upcard",
            'score_areas': "Score display areas"
        }

        self.regions = {}

        for region_name, description in regions_to_calibrate.items():
            print(f"\nCalibrating: {region_name}")
            print(f"Description: {description}")
            print("Click and drag to select the region on your screen")
            print("Press Enter when ready...")

            input()

            region = self._capture_region_interactive()
            if region:
                self.regions[region_name] = region
                print(f"✓ Calibrated {region_name}: {region}")
            else:
                print(f"✗ Failed to calibrate {region_name}")
                self.regions[region_name] = self._get_default_blackjack_region(region_name)

        return self.regions

    def _capture_region_interactive(self) -> Optional[Tuple[int, int, int, int]]:
        """Capture a region interactively from screen"""
        try:
            print("Opening screen capture window...")
            print("Use your mouse to select the region:")
            print("- Click and drag to select area")
            print("- Press ESC or 'q' to cancel")

            # Create a full screen window to show the selection
            screen = self._get_screen_capture()

            if screen is None:
                print("Failed to capture screen")
                return None

            # Create window to show screen capture
            cv2.namedWindow('Screen Capture - Select Region', cv2.WINDOW_NORMAL)
            cv2.setWindowProperty('Screen Capture - Select Region', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

            # Variables for selection
            self.calibration_points = []
            self.current_region = None

            # Set mouse callback
            cv2.setMouseCallback('Screen Capture - Select Region', self._mouse_callback)

            while True:
                display_image = screen.copy()

                # Draw selection rectangle if points exist
                if len(self.calibration_points) == 2:
                    cv2.rectangle(display_image,
                                self.calibration_points[0],
                                self.calibration_points[1],
                                (0, 255, 0), 2)

                cv2.imshow('Screen Capture - Select Region', display_image)

                key = cv2.waitKey(1) & 0xFF

                if key == 27 or key == ord('q'):  # ESC or 'q' to cancel
                    cv2.destroyWindow('Screen Capture - Select Region')
                    return None
                elif key == 13 and len(self.calibration_points) == 2:  # Enter to confirm
                    region = self._points_to_region(self.calibration_points[0], self.calibration_points[1])
                    cv2.destroyWindow('Screen Capture - Select Region')
                    return region

            cv2.destroyWindow('Screen Capture - Select Region')
            return None

        except Exception as e:
            print(f"Error in interactive capture: {e}")
            return None

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.calibration_points = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            if len(self.calibration_points) == 1:
                self.calibration_points.append((x, y))

    def _points_to_region(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Convert two points to a region (x, y, width, height)"""
        x1, y1 = p1
        x2, y2 = p2

        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        return (x, y, width, height)

    def _get_screen_capture(self) -> Optional[np.ndarray]:
        """Get screen capture using mss"""
        try:
            import mss
            sct = mss.mss()
            monitor = sct.monitors[1]  # Primary monitor
            screenshot = sct.grab(monitor)

            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception as e:
            print(f"Screen capture failed: {e}")
            return None

    def _get_default_poker_region(self, region_name: str) -> Tuple[int, int, int, int]:
        """Get default poker regions for fallback"""
        defaults = {
            'player_cards': (100, 400, 200, 100),
            'community_cards': (300, 300, 400, 100),
            'pot_info': (500, 100, 200, 50),
            'player_info': (50, 500, 150, 100)
        }
        return defaults.get(region_name, (0, 0, 100, 100))

    def _get_default_blackjack_region(self, region_name: str) -> Tuple[int, int, int, int]:
        """Get default blackjack regions for fallback"""
        defaults = {
            'player_cards': (200, 400, 300, 100),
            'dealer_cards': (200, 200, 300, 100),
            'dealer_upcard': (200, 200, 150, 100),
            'score_areas': (500, 200, 100, 200)
        }
        return defaults.get(region_name, (0, 0, 100, 100))


def run_calibration():
    """Run the calibration process"""
    calibrator = RegionCalibrator()

    print("Poker/Blackjack AI Agent - Interactive Calibration")
    print("=" * 60)
    print("This tool will help you calibrate the game detection regions")
    print("for better accuracy in poker and blackjack detection.")
    print()

    # Calibrate poker regions
    poker_regions = calibrator.calibrate_poker_regions()

    print("\n" + "=" * 60)
    print("Poker regions calibrated:")
    for name, region in poker_regions.items():
        print(f"  {name}: {region}")

    # Calibrate blackjack regions
    blackjack_regions = calibrator.calibrate_blackjack_regions()

    print("\n" + "=" * 60)
    print("Blackjack regions calibrated:")
    for name, region in blackjack_regions.items():
        print(f"  {name}: {region}")

    # Save calibration results
    save_calibration(poker_regions, blackjack_regions)

    print("\n✓ Calibration completed and saved!")
    print("You can now run the AI agent with improved detection accuracy.")


def save_calibration(poker_regions: Dict[str, Tuple], blackjack_regions: Dict[str, Tuple]):
    """Save calibration results to config file"""
    try:
        # Update the config
        config.config.game.poker_regions.update(poker_regions)
        config.config.game.blackjack_regions.update(blackjack_regions)

        print("Calibration data saved to configuration.")

    except Exception as e:
        print(f"Warning: Could not save calibration to config: {e}")


if __name__ == "__main__":
    run_calibration()