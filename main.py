#!/usr/bin/env python3
"""
Poker/Blackjack AI Agent - Main Entry Point
An AI agent that monitors your screen and provides real-time odds for poker and blackjack games.
"""
import sys
import time
import argparse
import threading
import signal
import atexit
from src.ai_agent import AIAgentManager
import config.config as config


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    print(f"\nReceived signal {signum}, shutting down gracefully...")
    if 'manager' in globals():
        manager.stop_agent()
    sys.exit(0)


def cleanup():
    """Cleanup function called on exit"""
    print("Cleaning up...")
    if 'manager' in globals():
        manager.stop_agent()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Poker/Blackjack AI Odds Agent')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run calibration mode to set up game regions')
    parser.add_argument('--debug', action='store_true',
                       help='Save debug screenshots during operation')
    parser.add_argument('--overlay-x', type=int, default=10,
                       help='Overlay X position (default: 10)')
    parser.add_argument('--overlay-y', type=int, default=10,
                       help='Overlay Y position (default: 10)')
    parser.add_argument('--transparency', type=float, default=0.9,
                       help='Overlay transparency 0.0-1.0 (default: 0.9)')

    args = parser.parse_args()

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup)

    try:
        # Initialize agent manager
        manager = AIAgentManager()

        # Set overlay configuration
        manager.agent.set_overlay_position(args.overlay_x, args.overlay_y)
        manager.agent.set_overlay_transparency(args.transparency)

        if args.calibrate:
            print("Starting calibration mode...")
            print("Click on the corners of the game regions to calibrate detection areas.")
            regions = manager.calibrate_regions()
            print(f"Calibrated regions: {regions}")
            return

        # Start the agent
        print("Starting Poker/Blackjack AI Agent...")
        print("The agent will monitor your screen for poker and blackjack games.")
        print("An overlay window will display real-time odds and recommendations.")
        print("Press Ctrl+C to stop the agent.")

        if config.config.ui.enable_hotkeys:
            print(f"Press {config.config.ui.toggle_hotkey} to toggle agent on/off")
            print(f"Press {config.config.ui.pause_hotkey} to pause/resume detection")

        if not manager.start_agent():
            print("Failed to start agent")
            return 1

        # Save debug screenshots if requested
        if args.debug:
            def save_debug_screenshot():
                timestamp = int(time.time())
                filename = f"debug_{timestamp}.png"
                manager.save_debug_screenshot(filename)

            # Save a screenshot every 30 seconds for debugging
            def debug_loop():
                while True:
                    time.sleep(30)
                    save_debug_screenshot()

            debug_thread = threading.Thread(target=debug_loop, daemon=True)
            debug_thread.start()

        # Start the overlay mainloop in the main thread
        try:
            # Print status periodically
            def status_loop():
                while True:
                    time.sleep(30)
                    status = manager.get_agent_status()
                    enabled_str = "ENABLED" if status['enabled'] else "DISABLED"
                    paused_str = "PAUSED" if status['paused'] else "ACTIVE"
                    print(f"Status - {enabled_str}, {paused_str}, "
                          f"Game: {status['current_game'] or 'None'}, "
                          f"Confidence: {status['detection_confidence']:.1%}")

            status_thread = threading.Thread(target=status_loop, daemon=True)
            status_thread.start()

            # Start the GUI mainloop in the main thread
            manager.agent.overlay_manager.start_mainloop()

        except KeyboardInterrupt:
            print("\nStopping agent...")
            manager.stop_agent()
            print("Agent stopped")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())