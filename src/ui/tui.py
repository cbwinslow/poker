"""
Terminal User Interface for Poker/Blackjack AI Agent
Provides real-time dashboard and control interface
"""
import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import deque
import curses
import textwrap
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.align import Align
import config.config as config


@dataclass
class OddsHistory:
    """Historical odds data point"""
    timestamp: datetime
    game_type: str
    game_state: Dict[str, Any]
    odds_data: Dict[str, Any]
    confidence: float


class Dashboard:
    """Real-time dashboard for displaying game state and odds"""

    def __init__(self):
        self.console = Console()
        self.history: deque[OddsHistory] = deque(maxlen=1000)  # Keep last 1000 entries
        self.current_game_state = None
        self.current_odds = None
        self.is_running = False
        self.layout = self._create_layout()

    def _create_layout(self) -> Layout:
        """Create the dashboard layout"""
        layout = Layout()

        # Split into header, main content, and footer
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )

        # Split main into game state and odds
        layout["main"].split_row(
            Layout(name="game_state"),
            Layout(name="odds_history")
        )

        return layout

    def update_game_state(self, game_type: str, game_state: Dict[str, Any], odds_data: Dict[str, Any], confidence: float):
        """Update current game state and add to history"""
        history_entry = OddsHistory(
            timestamp=datetime.now(),
            game_type=game_type,
            game_state=game_state,
            odds_data=odds_data,
            confidence=confidence
        )

        self.history.append(history_entry)
        self.current_game_state = game_state
        self.current_odds = odds_data

    def render_header(self) -> Panel:
        """Render header with status and controls"""
        status = "🟢 RUNNING" if self.is_running else "🔴 STOPPED"
        game_type = self.current_game_state.get('game_type', 'None') if self.current_game_state else 'None'

        current_time = datetime.now().strftime('%H:%M:%S')
        history_count = len(self.history)

        header_text = f"""
[bold blue]Poker/Blackjack AI Agent Dashboard[/bold blue]
Status: {status} | Current Game: {game_type} | History: {history_count} entries
Time: {current_time}
        """

        return Panel(header_text.strip(), title="🎯 AI Agent Control", border_style="blue")

    def render_game_state(self) -> Panel:
        """Render current game state"""
        if not self.current_game_state:
            content = "[dim]Waiting for game detection...[/dim]"
        else:
            game_type = self.current_game_state.get('game_type', 'Unknown')

            if game_type == 'poker':
                content = self._render_poker_state()
            elif game_type == 'blackjack':
                content = self._render_blackjack_state()
            else:
                content = f"[yellow]Unknown game type: {game_type}[/yellow]"

        return Panel(content, title="🎲 Current Game State", border_style="green")

    def _render_poker_state(self) -> str:
        """Render poker game state"""
        state = self.current_game_state
        player_cards = ', '.join([f"{card.get('rank', '?')} of {card.get('suit', 'unknown')}" for card in state.get('player_cards', [])])
        community_cards = ', '.join([f"{card.get('rank', '?')} of {card.get('suit', 'unknown')}" for card in state.get('community_cards', [])])

        game_phase = state.get('game_phase', 'Unknown').title()
        pot_size = state.get('pot_size', 0)
        confidence = state.get('confidence', 0)

        content = f"""
[bold]Poker Game[/bold]

[blue]Player Cards:[/blue]
{player_cards}

[blue]Community Cards:[/blue]
{community_cards}

[blue]Game Phase:[/blue] {game_phase}
[blue]Pot Size:[/blue] ${pot_size}
[blue]Confidence:[/blue] {confidence:.1%}
        """
        return content.strip()

    def _render_blackjack_state(self) -> str:
        """Render blackjack game state"""
        state = self.current_game_state
        player_cards = ', '.join([f"{card.get('rank', '?')} of {card.get('suit', 'unknown')} ({card.get('value', 0)})" for card in state.get('player_cards', [])])
        dealer_cards = ', '.join([f"{card.get('rank', '?')} of {card.get('suit', 'unknown')} ({card.get('value', 0)})" for card in state.get('dealer_cards', [])])
        player_score = state.get('player_score', 0)
        dealer_score = state.get('dealer_score', 0)
        confidence = state.get('confidence', 0)

        content = f"""
[bold]Blackjack Game[/bold]

[blue]Player Cards:[/blue]
{player_cards}

[blue]Dealer Cards:[/blue]
{dealer_cards}

[blue]Player Score:[/blue] {player_score}
[blue]Dealer Score:[/blue] {dealer_score}
[blue]Confidence:[/blue] {confidence:.1%}
        """
        return content.strip()

    def render_odds_display(self) -> Panel:
        """Render current odds and recommendations"""
        if not self.current_odds:
            content = "[dim]No odds data available[/dim]"
        else:
            game_type = self.current_odds.get('game_type', 'Unknown')

            if game_type == 'poker':
                content = self._render_poker_odds()
            elif game_type == 'blackjack':
                content = self._render_blackjack_odds()
            else:
                content = "[yellow]Unknown odds type[/yellow]"

        return Panel(content, title="📊 Real-time Odds", border_style="yellow")

    def _render_poker_odds(self) -> str:
        """Render poker odds"""
        odds = self.current_odds
        hand_category = odds.get('hand_category', 'Unknown').title()
        equity = odds.get('equity_vs_random', 0)
        pot_odds = odds.get('pot_odds', 0)
        action = odds.get('recommended_action', 'Wait').upper()
        outs = odds.get('outs', 0)

        content = f"""
[bold]Poker Odds Analysis[/bold]

[green]Hand Strength:[/green] {hand_category}
[green]Win Probability:[/green] {equity".1%"}
[green]Pot Odds:[/green] {pot_odds".1%"}
[green]Recommended Action:[/green] [bold]{action}[/bold]
[green]Outs:[/green] {outs}
        """
        return content.strip()

    def _render_blackjack_odds(self) -> str:
        """Render blackjack odds"""
        odds = self.current_odds
        win_prob = odds.get('player_win_probability', 0)
        bust_prob = odds.get('dealer_bust_probability', 0)
        true_count = odds.get('true_count', 0)
        action = odds.get('recommended_action', 'Wait').upper()

        content = f"""
[bold]Blackjack Odds Analysis[/bold]

[green]Win Probability:[/green] {win_prob".1%"}
[green]Dealer Bust %:[/green] {bust_prob".1%"}
[green]True Count:[/green] {true_count".1f"}
[green]Recommended Action:[/green] [bold]{action}[/bold]
        """
        return content.strip()

    def render_history(self) -> Panel:
        """Render odds history"""
        if not self.history:
            content = "[dim]No history available[/dim]"
        else:
            table = Table(title="📈 Recent History")
            table.add_column("Time", style="cyan", width=8)
            table.add_column("Game", style="magenta", width=10)
            table.add_column("State", style="green", width=20)
            table.add_column("Odds", style="yellow", width=20)
            table.add_column("Conf", style="red", width=6)

            # Show last 10 entries
            for entry in list(self.history)[-10:]:
                table.add_row(
                    entry.timestamp.strftime("%H:%M:%S"),
                    entry.game_type.title(),
                    self._truncate_text(str(entry.game_state), 18),
                    self._truncate_text(str(entry.odds_data), 18),
                    f"{entry.confidence".1%"}"
                )

            content = table

        return Panel(content, title="📚 Odds History", border_style="magenta")

    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to max length"""
        if len(text) <= max_length:
            return text
        return text[:max_length-3] + "..."

    def render_footer(self) -> Panel:
        """Render footer with controls"""
        controls = """
[bold]Controls:[/bold]
[cyan]SPACE[/cyan] - Toggle Agent | [cyan]H[/cyan] - Show/Hide History | [cyan]C[/cyan] - Clear History | [cyan]Q[/cyan] - Quit
        """
        return Panel(controls.strip(), title="⌨️  Controls", border_style="white")

    def render(self) -> Layout:
        """Render complete dashboard"""
        self.layout["header"].update(self.render_header())
        self.layout["game_state"].update(self.render_game_state())
        self.layout["odds_history"].update(self.render_odds_display())
        self.layout["footer"].update(self.render_footer())

        return self.layout

    def toggle_running(self):
        """Toggle agent running state"""
        self.is_running = not self.is_running

    def clear_history(self):
        """Clear odds history"""
        self.history.clear()

    def export_history(self, filename: str = None) -> str:
        """Export history to JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"odds_history_{timestamp}.json"

        # Convert history to JSON-serializable format
        export_data = []
        for entry in self.history:
            export_data.append({
                'timestamp': entry.timestamp.isoformat(),
                'game_type': entry.game_type,
                'game_state': entry.game_state,
                'odds_data': entry.odds_data,
                'confidence': entry.confidence
            })

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        return filename


class TUI:
    """Terminal User Interface"""

    def __init__(self):
        self.dashboard = Dashboard()
        self.console = Console()
        self.running = False

    def start(self):
        """Start the TUI"""
        self.running = True

        # Start dashboard update thread
        update_thread = threading.Thread(target=self._update_loop, daemon=True)
        update_thread.start()

        try:
            with Live(self.dashboard.render(), refresh_per_second=4, screen=True) as live:
                while self.running:
                    # Handle keyboard input
                    if self.console.is_terminal:
                        # In a real implementation, you'd handle keyboard input here
                        # For now, just update the display
                        time.sleep(0.1)

        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the TUI"""
        self.running = False

    def _update_loop(self):
        """Background update loop"""
        while self.running:
            # Update dashboard (in real implementation, this would get data from the AI agent)
            time.sleep(0.25)

    def update_game_data(self, game_type: str, game_state: Dict[str, Any], odds_data: Dict[str, Any], confidence: float):
        """Update game data from AI agent"""
        self.dashboard.update_game_state(game_type, game_state, odds_data, confidence)

    def handle_input(self, key: str):
        """Handle user input"""
        if key == ' ' or key == 'SPACE':
            self.dashboard.toggle_running()
        elif key.lower() == 'h':
            # Toggle history view (simplified)
            pass
        elif key.lower() == 'c':
            self.dashboard.clear_history()
        elif key.lower() == 'q':
            self.stop()
        elif key.lower() == 'e':
            # Export history
            filename = self.dashboard.export_history()
            self.console.print(f"[green]History exported to: {filename}[/green]")


# Global TUI instance
tui = TUI()


def start_tui():
    """Start the TUI (can be called from main)"""
    tui.start()


def update_tui_data(game_type: str, game_state: Dict[str, Any], odds_data: Dict[str, Any], confidence: float):
    """Update TUI with new game data (called from AI agent)"""
    tui.update_game_data(game_type, game_state, odds_data, confidence)


def stop_tui():
    """Stop the TUI"""
    tui.stop()
