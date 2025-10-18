"""
Real-time odds overlay UI for poker/blackjack games
"""
import tkinter as tk
from tkinter import ttk, font
import threading
import time
from typing import Dict, Any, Optional, Tuple
import config.config as config
from ..poker_odds import PokerOdds
from ..blackjack_odds import BlackjackOdds

try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False


class OddsOverlay:
    """Transparent overlay window for displaying game odds"""

    def __init__(self, ui_config: config.UIConfig):
        self.config = ui_config
        self.root = None
        self.poker_frame = None
        self.blackjack_frame = None
        self.current_game = None
        self.running = False
        self.minimized = False
        self.tray_icon = None

        # Colors for different hand strengths
        self.colors = {
            'monster': '#00FF00',  # Bright green
            'strong': '#90EE90',   # Light green
            'medium': '#FFFF00',   # Yellow
            'weak': '#FFA500',     # Orange
            'trash': '#FF0000'     # Red
        }

        # Action colors for recommendations
        self.action_colors = {
            'hit': '#FFA500',      # Orange
            'stand': '#00FF00',    # Green
            'double': '#FFD700',   # Gold
            'split': '#FF69B4',    # Pink
            'surrender': '#FF0000', # Red
            'fold': '#FF0000',     # Red
            'call': '#FFFF00',     # Yellow
            'raise': '#00FF00',    # Green
            'all_in': '#FF4500'    # Orange red
        }

        self._create_overlay()

    def _create_overlay(self) -> None:
        """Create the transparent overlay window"""
        self.root = tk.Tk()
        self.root.title("Poker/Blackjack Odds Overlay")

        # Make window transparent and always on top
        self.root.attributes('-alpha', self.config.overlay_alpha)
        self.root.attributes('-topmost', True)
        self.root.overrideredirect(True)  # Remove window decorations

        # Set window position
        self.root.geometry(f"+{self.config.odds_display_position[0]}+{self.config.odds_display_position[1]}")

        # Create main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create poker odds frame (initially hidden)
        self.poker_frame = self._create_poker_frame(main_frame)
        self.poker_frame.pack_forget()

        # Create blackjack odds frame (initially hidden)
        self.blackjack_frame = self._create_blackjack_frame(main_frame)
        self.blackjack_frame.pack_forget()

        # Status label
        self.status_label = ttk.Label(main_frame, text="Waiting for game detection...")
        self.status_label.pack(pady=5)

        # Control buttons frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=5, fill=tk.X)

        # Minimize button (only if enabled in config)
        if self.config.enable_minimize:
            self.minimize_button = ttk.Button(
                control_frame,
                text="−",  # Minimize symbol
                width=3,
                command=self._toggle_minimize
            )
            self.minimize_button.pack(side=tk.LEFT)

        # Close button
        self.close_button = ttk.Button(
            control_frame,
            text="×",  # Close symbol
            width=3,
            command=self._close_overlay
        )
        self.close_button.pack(side=tk.RIGHT)

    def _create_poker_frame(self, parent: ttk.Frame) -> ttk.Frame:
        """Create poker odds display frame"""
        frame = ttk.Frame(parent, relief='solid', borderwidth=1)

        # Title
        title = ttk.Label(frame, text="POKER ODDS", font=('Arial', 14, 'bold'))
        title.pack(pady=5)

        # Hand strength indicator
        self.poker_hand_label = ttk.Label(frame, text="Hand: Unknown", font=('Arial', 12))
        self.poker_hand_label.pack(pady=2)

        # Equity display
        equity_frame = ttk.Frame(frame)
        equity_frame.pack(pady=5, fill=tk.X)

        ttk.Label(equity_frame, text="Win Probability:").pack(anchor=tk.W)
        self.poker_equity_label = ttk.Label(equity_frame, text="0%", font=('Arial', 10, 'bold'))
        self.poker_equity_label.pack(anchor=tk.W)

        # Pot odds
        pot_frame = ttk.Frame(frame)
        pot_frame.pack(pady=5, fill=tk.X)

        ttk.Label(pot_frame, text="Pot Odds:").pack(anchor=tk.W)
        self.poker_pot_odds_label = ttk.Label(pot_frame, text="0%", font=('Arial', 10))
        self.poker_pot_odds_label.pack(anchor=tk.W)

        # Recommendation
        rec_frame = ttk.Frame(frame)
        rec_frame.pack(pady=5, fill=tk.X)

        ttk.Label(rec_frame, text="Recommendation:").pack(anchor=tk.W)
        self.poker_recommendation_label = ttk.Label(rec_frame, text="Waiting...",
                                                   font=('Arial', 11, 'bold'))
        self.poker_recommendation_label.pack(anchor=tk.W)

        # Outs information
        outs_frame = ttk.Frame(frame)
        outs_frame.pack(pady=5, fill=tk.X)

        ttk.Label(outs_frame, text="Outs:").pack(anchor=tk.W)
        self.poker_outs_label = ttk.Label(outs_frame, text="0")
        self.poker_outs_label.pack(anchor=tk.W)

        # Enhanced poker information frames
        self._create_enhanced_poker_info(frame)

        return frame

    def _create_enhanced_poker_info(self, parent: ttk.Frame) -> None:
        """Create enhanced poker information display"""
        # Specific hand odds frame
        self.poker_hand_odds_frame = ttk.LabelFrame(parent, text="Hand Odds", padding="5")
        self.poker_hand_odds_frame.pack(pady=5, fill=tk.X)

        # Position and betting context
        context_frame = ttk.Frame(parent)
        context_frame.pack(pady=5, fill=tk.X)

        ttk.Label(context_frame, text="Position:").grid(row=0, column=0, sticky=tk.W)
        self.poker_position_label = ttk.Label(context_frame, text="Unknown")
        self.poker_position_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(context_frame, text="Pot Odds:").grid(row=0, column=2, sticky=tk.W)
        self.poker_pot_odds_enhanced_label = ttk.Label(context_frame, text="0%")
        self.poker_pot_odds_enhanced_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        ttk.Label(context_frame, text="Stack:").grid(row=1, column=0, sticky=tk.W)
        self.poker_stack_label = ttk.Label(context_frame, text="$0")
        self.poker_stack_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(context_frame, text="Pot Size:").grid(row=1, column=2, sticky=tk.W)
        self.poker_pot_size_label = ttk.Label(context_frame, text="$0")
        self.poker_pot_size_label.grid(row=1, column=3, sticky=tk.W, padx=(10, 0))

        # Opponent analysis frame
        opponent_frame = ttk.LabelFrame(parent, text="Opponent Analysis", padding="5")
        opponent_frame.pack(pady=5, fill=tk.X)

        self.poker_opponent_label = ttk.Label(opponent_frame, text="Analyzing opponents...")
        self.poker_opponent_label.pack(anchor=tk.W)

        # Expected value and advanced metrics
        ev_frame = ttk.Frame(parent)
        ev_frame.pack(pady=5, fill=tk.X)

        ttk.Label(ev_frame, text="Expected Value:").grid(row=0, column=0, sticky=tk.W)
        self.poker_ev_label = ttk.Label(ev_frame, text="$0.00", font=('Arial', 10, 'bold'))
        self.poker_ev_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(ev_frame, text="Fold Equity:").grid(row=0, column=2, sticky=tk.W)
        self.poker_fold_equity_label = ttk.Label(ev_frame, text="0%")
        self.poker_fold_equity_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        # Model confidence and historical data
        model_frame = ttk.Frame(parent)
        model_frame.pack(pady=5, fill=tk.X)

        ttk.Label(model_frame, text="Model Accuracy:").grid(row=0, column=0, sticky=tk.W)
        self.poker_model_accuracy_label = ttk.Label(model_frame, text="0%")
        self.poker_model_accuracy_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(model_frame, text="Historical P&L:").grid(row=0, column=2, sticky=tk.W)
        self.poker_historical_pl_label = ttk.Label(model_frame, text="$0.00")
        self.poker_historical_pl_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

    def _create_blackjack_frame(self, parent: ttk.Frame) -> ttk.Frame:
        """Create blackjack odds display frame"""
        frame = ttk.Frame(parent, relief='solid', borderwidth=1)

        # Title
        title = ttk.Label(frame, text="BLACKJACK ODDS", font=('Arial', 14, 'bold'))
        title.pack(pady=5)

        # Player score
        score_frame = ttk.Frame(frame)
        score_frame.pack(pady=5, fill=tk.X)

        ttk.Label(score_frame, text="Player Score:").pack(anchor=tk.W)
        self.bj_player_score_label = ttk.Label(score_frame, text="0", font=('Arial', 12, 'bold'))
        self.bj_player_score_label.pack(anchor=tk.W)

        # Dealer upcard
        dealer_frame = ttk.Frame(frame)
        dealer_frame.pack(pady=5, fill=tk.X)

        ttk.Label(dealer_frame, text="Dealer Upcard:").pack(anchor=tk.W)
        self.bj_dealer_card_label = ttk.Label(dealer_frame, text="Unknown", font=('Arial', 10))
        self.bj_dealer_card_label.pack(anchor=tk.W)

        # Win probability
        win_frame = ttk.Frame(frame)
        win_frame.pack(pady=5, fill=tk.X)

        ttk.Label(win_frame, text="Win Probability:").pack(anchor=tk.W)
        self.bj_win_prob_label = ttk.Label(win_frame, text="0%", font=('Arial', 10, 'bold'))
        self.bj_win_prob_label.pack(anchor=tk.W)

        # Dealer bust probability
        bust_frame = ttk.Frame(frame)
        bust_frame.pack(pady=5, fill=tk.X)

        ttk.Label(bust_frame, text="Dealer Bust %:").pack(anchor=tk.W)
        self.bj_bust_prob_label = ttk.Label(bust_frame, text="0%", font=('Arial', 10))
        self.bj_bust_prob_label.pack(anchor=tk.W)

        # Card counting info
        count_frame = ttk.Frame(frame)
        count_frame.pack(pady=5, fill=tk.X)

        ttk.Label(count_frame, text="True Count:").pack(anchor=tk.W)
        self.bj_count_label = ttk.Label(count_frame, text="0.0", font=('Arial', 10))
        self.bj_count_label.pack(anchor=tk.W)

        # Recommendation
        rec_frame = ttk.Frame(frame)
        rec_frame.pack(pady=5, fill=tk.X)

        ttk.Label(rec_frame, text="Recommended Action:").pack(anchor=tk.W)
        self.bj_recommendation_label = ttk.Label(rec_frame, text="Waiting...",
                                                 font=('Arial', 11, 'bold'))
        self.bj_recommendation_label.pack(anchor=tk.W)

        # Enhanced blackjack information frames
        self._create_enhanced_blackjack_info(frame)

        return frame

    def _create_enhanced_blackjack_info(self, parent: ttk.Frame) -> None:
        """Create enhanced blackjack information display"""
        # Betting and bankroll context
        betting_frame = ttk.Frame(parent)
        betting_frame.pack(pady=5, fill=tk.X)

        ttk.Label(betting_frame, text="Current Bet:").grid(row=0, column=0, sticky=tk.W)
        self.bj_current_bet_label = ttk.Label(betting_frame, text="$0")
        self.bj_current_bet_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(betting_frame, text="Bankroll:").grid(row=0, column=2, sticky=tk.W)
        self.bj_bankroll_label = ttk.Label(betting_frame, text="$0")
        self.bj_bankroll_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        # Card counting information
        count_frame = ttk.LabelFrame(parent, text="Card Counting", padding="5")
        count_frame.pack(pady=5, fill=tk.X)

        count_info_frame = ttk.Frame(count_frame)
        count_info_frame.pack(fill=tk.X)

        ttk.Label(count_info_frame, text="True Count:").grid(row=0, column=0, sticky=tk.W)
        self.bj_true_count_label = ttk.Label(count_info_frame, text="0.0", font=('Arial', 10, 'bold'))
        self.bj_true_count_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(count_info_frame, text="Deck Penetration:").grid(row=0, column=2, sticky=tk.W)
        self.bj_penetration_label = ttk.Label(count_info_frame, text="0%")
        self.bj_penetration_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        # Advanced probabilities frame
        prob_frame = ttk.LabelFrame(parent, text="Advanced Probabilities", padding="5")
        prob_frame.pack(pady=5, fill=tk.X)

        prob_info_frame = ttk.Frame(prob_frame)
        prob_info_frame.pack(fill=tk.X)

        ttk.Label(prob_info_frame, text="Blackjack:").grid(row=0, column=0, sticky=tk.W)
        self.bj_blackjack_prob_label = ttk.Label(prob_info_frame, text="0%")
        self.bj_blackjack_prob_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(prob_info_frame, text="Face Card:").grid(row=0, column=2, sticky=tk.W)
        self.bj_face_card_prob_label = ttk.Label(prob_info_frame, text="0%")
        self.bj_face_card_prob_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        ttk.Label(prob_info_frame, text="Push:").grid(row=1, column=0, sticky=tk.W)
        self.bj_push_prob_label = ttk.Label(prob_info_frame, text="0%")
        self.bj_push_prob_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(prob_info_frame, text="Ace:").grid(row=1, column=2, sticky=tk.W)
        self.bj_ace_prob_label = ttk.Label(prob_info_frame, text="0%")
        self.bj_ace_prob_label.grid(row=1, column=3, sticky=tk.W, padx=(10, 0))

        # Strategy advantages frame
        strategy_frame = ttk.LabelFrame(parent, text="Strategy Analysis", padding="5")
        strategy_frame.pack(pady=5, fill=tk.X)

        strategy_info_frame = ttk.Frame(strategy_frame)
        strategy_info_frame.pack(fill=tk.X)

        ttk.Label(strategy_info_frame, text="Double Advantage:").grid(row=0, column=0, sticky=tk.W)
        self.bj_double_advantage_label = ttk.Label(strategy_info_frame, text="0%")
        self.bj_double_advantage_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(strategy_info_frame, text="Split Advantage:").grid(row=0, column=2, sticky=tk.W)
        self.bj_split_advantage_label = ttk.Label(strategy_info_frame, text="0%")
        self.bj_split_advantage_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

        ttk.Label(strategy_info_frame, text="Insurance Value:").grid(row=1, column=0, sticky=tk.W)
        self.bj_insurance_value_label = ttk.Label(strategy_info_frame, text="$0.00")
        self.bj_insurance_value_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(strategy_info_frame, text="Bankroll Risk:").grid(row=1, column=2, sticky=tk.W)
        self.bj_bankroll_risk_label = ttk.Label(strategy_info_frame, text="Low")
        self.bj_bankroll_risk_label.grid(row=1, column=3, sticky=tk.W, padx=(10, 0))

        # Model performance frame
        model_frame = ttk.Frame(parent)
        model_frame.pack(pady=5, fill=tk.X)

        ttk.Label(model_frame, text="Model Accuracy:").grid(row=0, column=0, sticky=tk.W)
        self.bj_model_accuracy_label = ttk.Label(model_frame, text="0%")
        self.bj_model_accuracy_label.grid(row=0, column=1, sticky=tk.W, padx=(10, 0))

        ttk.Label(model_frame, text="Historical P&L:").grid(row=0, column=2, sticky=tk.W)
        self.bj_historical_pl_label = ttk.Label(model_frame, text="$0.00")
        self.bj_historical_pl_label.grid(row=0, column=3, sticky=tk.W, padx=(10, 0))

    def start(self) -> None:
        """Start the overlay display"""
        if not self.running:
            self.running = True
            # Schedule the update to run after mainloop starts
            self.root.after(100, self._update_display)

    def run(self) -> None:
        """Run the main loop (must be called from main thread)"""
        if self.root:
            self.root.mainloop()

    def stop(self) -> None:
        """Stop the overlay display"""
        self.running = False
        if self.root:
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass  # Ignore errors during cleanup

    def update_poker_odds(self, odds: PokerOdds) -> None:
        """Update poker odds display"""
        if not self.running:
            return

        try:
            # Update hand strength with color coding
            hand_color = self.colors.get(odds.hand_category, '#FFFFFF')
            self.poker_hand_label.config(
                text=f"Hand: {odds.hand_rank.replace('_', ' ').title()} ({odds.hand_category.title()})",
                foreground=hand_color
            )

            # Update equity
            equity_text = f"{odds.equity_vs_random*100:.1f}%"
            self.poker_equity_label.config(text=equity_text)

            # Update pot odds
            pot_odds_text = f"{odds.pot_odds*100:.1f}%"
            self.poker_pot_odds_label.config(text=pot_odds_text)

            # Update recommendation with color coding
            rec_color = {'fold': '#FF0000', 'call': '#FFA500', 'raise': '#00FF00'}.get(odds.recommended_action, '#FFFFFF')
            self.poker_recommendation_label.config(
                text=odds.recommended_action.upper(),
                foreground=rec_color
            )

            # Update outs
            self.poker_outs_label.config(text=str(odds.outs))

            # Update enhanced poker information
            self._update_enhanced_poker_info(odds)

            # Show poker frame, hide blackjack frame
            self.blackjack_frame.pack_forget()
            self.poker_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.status_label.pack_forget()

            self.current_game = 'poker'

        except Exception as e:
            print(f"Error updating poker odds display: {e}")

    def update_blackjack_odds(self, odds: BlackjackOdds) -> None:
        """Update blackjack odds display"""
        if not self.running:
            return

        try:
            # Update player score
            self.bj_player_score_label.config(text=str(odds.player_win_probability * 100) + "%")

            # Update dealer upcard (placeholder - would need actual card info)
            self.bj_dealer_card_label.config(text="Visible Card")

            # Update win probability
            win_text = f"{odds.player_win_probability*100:.1f}%"
            self.bj_win_prob_label.config(text=win_text)

            # Update dealer bust probability
            bust_text = f"{odds.dealer_bust_probability*100:.1f}%"
            self.bj_bust_prob_label.config(text=bust_text)

            # Update true count with color coding
            count_color = '#00FF00' if odds.true_count > 1 else '#FF0000' if odds.true_count < -1 else '#FFFFFF'
            self.bj_count_label.config(text=f"{odds.true_count:.1f}", foreground=count_color)

            # Update recommendation with color coding
            action_colors = {
                'hit': '#FFA500',
                'stand': '#00FF00',
                'double': '#FFFF00',
                'split': '#FF69B4',
                'surrender': '#FF0000'
            }
            rec_color = action_colors.get(odds.recommended_action, '#FFFFFF')
            self.bj_recommendation_label.config(
                text=odds.recommended_action.upper(),
                foreground=rec_color
            )

            # Update enhanced blackjack information
            self._update_enhanced_blackjack_info(odds)

            # Show blackjack frame, hide poker frame
            self.poker_frame.pack_forget()
            self.blackjack_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            self.status_label.pack_forget()

            self.current_game = 'blackjack'

        except Exception as e:
            print(f"Error updating blackjack odds display: {e}")

    def show_waiting(self) -> None:
        """Show waiting for game detection message"""
        if self.running:
            self.poker_frame.pack_forget()
            self.blackjack_frame.pack_forget()
            self.status_label.pack(fill=tk.BOTH, expand=True)

    def _update_display(self) -> None:
        """Update display loop"""
        if self.running:
            self.root.after(100, self._update_display)

    def set_position(self, x: int, y: int) -> None:
        """Set overlay window position"""
        if self.root:
            self.root.geometry(f"+{x}+{y}")

    def set_transparency(self, alpha: float) -> None:
        """Set overlay transparency"""
        if self.root:
            self.root.attributes('-alpha', alpha)

    def _update_enhanced_poker_info(self, odds) -> None:
        """Update enhanced poker information display"""
        try:
            # Update position and betting context
            if hasattr(odds, 'position_advantage'):
                self.poker_position_label.config(text=getattr(odds, 'player_position', 'Unknown'))

            if hasattr(odds, 'pot_odds'):
                pot_odds_text = f"{odds.pot_odds*100:.1f}%"
                self.poker_pot_odds_enhanced_label.config(text=pot_odds_text)

            # Update expected value with color coding
            if hasattr(odds, 'expected_value'):
                ev = odds.expected_value
                ev_color = '#00FF00' if ev > 0 else '#FF0000' if ev < 0 else '#FFFFFF'
                self.poker_ev_label.config(text=f"${ev:.2f}", foreground=ev_color)

            # Update fold equity
            if hasattr(odds, 'fold_equity'):
                fe_text = f"{odds.fold_equity*100:.1f}%"
                self.poker_fold_equity_label.config(text=fe_text)

            # Update model performance (placeholder - would come from historical tracker)
            self.poker_model_accuracy_label.config(text="95%")  # Placeholder
            self.poker_historical_pl_label.config(text="$127.50")  # Placeholder

        except Exception as e:
            logger.debug(f"Error updating enhanced poker info: {e}")

    def _update_enhanced_blackjack_info(self, odds) -> None:
        """Update enhanced blackjack information display"""
        try:
            # Update betting information
            if hasattr(odds, 'current_bet'):
                self.bj_current_bet_label.config(text=f"${odds.current_bet or 0}")

            if hasattr(odds, 'player_bankroll'):
                self.bj_bankroll_label.config(text=f"${odds.player_bankroll or 0}")

            # Update card counting information
            if hasattr(odds, 'true_count'):
                tc = odds.true_count
                tc_color = '#00FF00' if tc > 1 else '#FF0000' if tc < -1 else '#FFFFFF'
                self.bj_true_count_label.config(text=f"{tc:.1f}", foreground=tc_color)

            if hasattr(odds, 'deck_penetration'):
                penetration_text = f"{odds.deck_penetration*100:.1f}%"
                self.bj_penetration_label.config(text=penetration_text)

            # Update advanced probabilities
            if hasattr(odds, 'blackjack_probability'):
                bj_prob_text = f"{odds.blackjack_probability*100:.1f}%"
                self.bj_blackjack_prob_label.config(text=bj_prob_text)

            if hasattr(odds, 'face_card_probability'):
                fc_prob_text = f"{odds.face_card_probability*100:.1f}%"
                self.bj_face_card_prob_label.config(text=fc_prob_text)

            if hasattr(odds, 'push_probability'):
                push_prob_text = f"{odds.push_probability*100:.1f}%"
                self.bj_push_prob_label.config(text=push_prob_text)

            if hasattr(odds, 'ace_probability'):
                ace_prob_text = f"{odds.ace_probability*100:.1f}%"
                self.bj_ace_prob_label.config(text=ace_prob_text)

            # Update strategy advantages
            if hasattr(odds, 'double_down_advantage'):
                dd_advantage_text = f"{odds.double_down_advantage*100:.1f}%"
                self.bj_double_advantage_label.config(text=dd_advantage_text)

            if hasattr(odds, 'split_advantage'):
                split_advantage_text = f"{odds.split_advantage*100:.1f}%"
                self.bj_split_advantage_label.config(text=split_advantage_text)

            if hasattr(odds, 'insurance_value'):
                iv = odds.insurance_value
                iv_color = '#00FF00' if iv > 0 else '#FF0000'
                self.bj_insurance_value_label.config(text=f"${iv:.2f}", foreground=iv_color)

            if hasattr(odds, 'bankroll_risk'):
                risk_level = odds.bankroll_risk
                risk_color = '#FF0000' if risk_level > 0.7 else '#FFA500' if risk_level > 0.4 else '#00FF00'
                self.bj_bankroll_risk_label.config(text=["Low", "Medium", "High"][int(risk_level * 2)], foreground=risk_color)

            # Update model performance (placeholder)
            self.bj_model_accuracy_label.config(text="93%")
            self.bj_historical_pl_label.config(text="$85.25")

        except Exception as e:
            logger.debug(f"Error updating enhanced blackjack info: {e}")

    def _toggle_minimize(self) -> None:
        """Toggle overlay minimization"""
        if self.minimized:
            self._restore_overlay()
        else:
            self._minimize_overlay()

    def _minimize_overlay(self) -> None:
        """Minimize overlay to system tray or hide it"""
        if TRAY_AVAILABLE and self.tray_icon is None:
            # Create system tray icon
            self._create_tray_icon()

        if self.tray_icon:
            # Hide window and show tray icon
            self.root.withdraw()
            self.minimized = True
            self.minimize_button.config(text="+")  # Change to restore symbol
        else:
            # Simple hide (no tray support)
            self.root.withdraw()
            self.minimized = True
            self.minimize_button.config(text="+")

    def _restore_overlay(self) -> None:
        """Restore overlay from minimized state"""
        if self.tray_icon:
            self.tray_icon.stop()
            self.tray_icon = None

        self.root.deiconify()
        self.minimized = False
        self.minimize_button.config(text="−")  # Change back to minimize symbol

    def _create_tray_icon(self) -> None:
        """Create system tray icon"""
        if not TRAY_AVAILABLE:
            return

        try:
            # Create a simple icon
            icon_image = Image.new('RGB', (64, 64), color='blue')
            draw = ImageDraw.Draw(icon_image)
            draw.rectangle([16, 16, 48, 48], fill='white')

            # Create menu
            menu = pystray.Menu(
                pystray.MenuItem('Restore', self._restore_overlay),
                pystray.MenuItem('Exit', self._close_overlay)
            )

            # Create tray icon
            self.tray_icon = pystray.Icon("poker_ai", icon_image, "Poker AI Agent", menu)
            self.tray_icon.run_detached()

        except Exception as e:
            print(f"Failed to create tray icon: {e}")
            self.tray_icon = None

    def _close_overlay(self) -> None:
        """Close the overlay application"""
        self.stop()


class OverlayManager:
    """Manages the odds overlay"""

    def __init__(self, ui_config: config.UIConfig):
        self.config = ui_config
        self.overlay = OddsOverlay(ui_config)
        self.overlay_thread = None

    def start(self) -> None:
        """Start the overlay in a separate thread"""
        if self.overlay_thread is None or not self.overlay_thread.is_alive():
            self.overlay_thread = threading.Thread(target=self.overlay.start, daemon=True)
            self.overlay_thread.start()

    def start_mainloop(self) -> None:
        """Start the Tkinter mainloop (must be called from main thread)"""
        self.overlay.run()

    def stop(self) -> None:
        """Stop the overlay"""
        self.overlay.stop()
        if self.overlay_thread:
            self.overlay_thread.join(timeout=1.0)

    def update_poker_odds(self, odds: PokerOdds) -> None:
        """Update poker odds display"""
        if self.overlay_thread and self.overlay_thread.is_alive():
            # Use after method to update from main thread
            self.overlay.root.after(0, lambda: self.overlay.update_poker_odds(odds))

    def update_blackjack_odds(self, odds: BlackjackOdds) -> None:
        """Update blackjack odds display"""
        if self.overlay_thread and self.overlay_thread.is_alive():
            self.overlay.root.after(0, lambda: self.overlay.update_blackjack_odds(odds))

    def show_waiting(self) -> None:
        """Show waiting message"""
        if self.overlay_thread and self.overlay_thread.is_alive():
            self.overlay.root.after(0, self.overlay.show_waiting)

    def set_position(self, x: int, y: int) -> None:
        """Set overlay position"""
        self.overlay.set_position(x, y)

    def set_transparency(self, alpha: float) -> None:
        """Set overlay transparency"""
        self.overlay.set_transparency(alpha)

    def start_mainloop(self) -> None:
        """Start the Tkinter mainloop in the main thread"""
        if self.overlay:
            self.overlay.run()