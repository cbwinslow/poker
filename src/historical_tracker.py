"""
Unified historical tracking system for poker and blackjack AI agent
Tracks actual vs expected results and improves models over time
"""
import json
import time
import sqlite3
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .poker_detector import PokerGameState
from .blackjack_detector import BlackjackGameState

logger = logging.getLogger(__name__)


@dataclass
class GameSession:
    """Represents a gaming session with multiple hands"""
    session_id: str
    game_type: str  # 'poker' or 'blackjack'
    start_time: float
    end_time: Optional[float] = None
    hands_played: int = 0
    total_profit_loss: float = 0.0
    location: Optional[str] = None


@dataclass
class HandResult:
    """Result of a single hand"""
    game_type: str
    timestamp: float
    predicted_action: str
    actual_outcome: str
    expected_value: float
    actual_profit_loss: float
    game_state_summary: Dict[str, Any]
    confidence_score: float
    model_version: str = "1.0"


class DatabaseManager:
    """SQLite database manager for historical data"""

    def __init__(self, db_path: str = 'ai_agent.db'):
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS gaming_sessions (
                    session_id TEXT PRIMARY KEY,
                    game_type TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL,
                    hands_played INTEGER DEFAULT 0,
                    total_profit_loss REAL DEFAULT 0.0,
                    location TEXT
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS hand_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    game_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    predicted_action TEXT,
                    actual_outcome TEXT,
                    expected_value REAL,
                    actual_profit_loss REAL,
                    confidence_score REAL,
                    model_version TEXT DEFAULT '1.0',
                    FOREIGN KEY (session_id) REFERENCES gaming_sessions (session_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS poker_hand_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hand_history_id INTEGER,
                    player_position TEXT,
                    game_phase TEXT,
                    pot_size REAL,
                    player_stack REAL,
                    hand_strength_category TEXT,
                    FOREIGN KEY (hand_history_id) REFERENCES hand_history (id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS blackjack_hand_details (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hand_history_id INTEGER,
                    player_score INTEGER,
                    dealer_upcard INTEGER,
                    true_count REAL,
                    current_bet REAL,
                    FOREIGN KEY (hand_history_id) REFERENCES hand_history (id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS opponent_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    opponent_id TEXT,
                    game_type TEXT NOT NULL,
                    hands_played INTEGER DEFAULT 0,
                    vpip REAL DEFAULT 0.0,  -- Voluntarily Put money In Pot %
                    pfr REAL DEFAULT 0.0,   -- Pre-Flop Raise %
                    af REAL DEFAULT 0.0,    -- Aggression Factor
                    total_profit_loss REAL DEFAULT 0.0,
                    last_seen REAL,
                    FOREIGN KEY (session_id) REFERENCES gaming_sessions (session_id)
                )
            ''')

            # Create indexes for better performance
            conn.execute('CREATE INDEX IF NOT EXISTS idx_hand_history_session ON hand_history(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_hand_history_game_type ON hand_history(game_type)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_hand_history_timestamp ON hand_history(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_opponent_stats_session ON opponent_statistics(session_id)')

    def insert_session(self, session: GameSession) -> None:
        """Insert or update gaming session"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO gaming_sessions
                (session_id, game_type, start_time, end_time, hands_played, total_profit_loss, location)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (session.session_id, session.game_type, session.start_time, session.end_time,
                  session.hands_played, session.total_profit_loss, session.location))

    def insert_hand_result(self, hand: HandResult) -> int:
        """Insert hand result and return the inserted ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO hand_history
                (session_id, game_type, timestamp, predicted_action, actual_outcome,
                 expected_value, actual_profit_loss, confidence_score, model_version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (None, hand.game_type, hand.timestamp, hand.predicted_action, hand.actual_outcome,
                  hand.expected_value, hand.actual_profit_loss, hand.confidence_score, hand.model_version))

            hand_id = cursor.lastrowid

            # Insert game-specific details
            if hand.game_type == 'poker' and 'player_position' in hand.game_state_summary:
                conn.execute('''
                    INSERT INTO poker_hand_details
                    (hand_history_id, player_position, game_phase, pot_size, player_stack, hand_strength_category)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (hand_id, hand.game_state_summary.get('player_position'),
                      hand.game_state_summary.get('game_phase'),
                      hand.game_state_summary.get('pot_size'),
                      hand.game_state_summary.get('player_stack'),
                      hand.game_state_summary.get('hand_strength_category')))

            elif hand.game_type == 'blackjack' and 'player_score' in hand.game_state_summary:
                conn.execute('''
                    INSERT INTO blackjack_hand_details
                    (hand_history_id, player_score, dealer_upcard, true_count, current_bet)
                    VALUES (?, ?, ?, ?, ?)
                ''', (hand_id, hand.game_state_summary.get('player_score'),
                      hand.game_state_summary.get('dealer_upcard'),
                      hand.game_state_summary.get('true_count'),
                      hand.game_state_summary.get('current_bet')))

            return hand_id

    def get_session_hands(self, session_id: str) -> List[HandResult]:
        """Get all hands for a specific session"""
        hands = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('''
                SELECT hh.*, phd.*, bhd.*
                FROM hand_history hh
                LEFT JOIN poker_hand_details phd ON hh.id = phd.hand_history_id
                LEFT JOIN blackjack_hand_details bhd ON hh.id = bhd.hand_history_id
                WHERE hh.session_id = ?
                ORDER BY hh.timestamp
            ''', (session_id,))

            for row in cursor:
                # Reconstruct game_state_summary
                game_state_summary = {}
                if row['player_position']:  # Poker hand
                    game_state_summary.update({
                        'player_position': row['player_position'],
                        'game_phase': row['game_phase'],
                        'pot_size': row['pot_size'],
                        'player_stack': row['player_stack'],
                        'hand_strength_category': row['hand_strength_category']
                    })
                else:  # Blackjack hand
                    game_state_summary.update({
                        'player_score': row['player_score'],
                        'dealer_upcard': row['dealer_upcard'],
                        'true_count': row['true_count'],
                        'current_bet': row['current_bet']
                    })

                hand = HandResult(
                    game_type=row['game_type'],
                    timestamp=row['timestamp'],
                    predicted_action=row['predicted_action'],
                    actual_outcome=row['actual_outcome'],
                    expected_value=row['expected_value'],
                    actual_profit_loss=row['actual_profit_loss'],
                    game_state_summary=game_state_summary,
                    confidence_score=row['confidence_score'],
                    model_version=row['model_version']
                )
                hands.append(hand)

        return hands

    def get_model_performance(self, game_type: str = None) -> Dict[str, Any]:
        """Get model performance statistics from database"""
        performance = {}

        with sqlite3.connect(self.db_path) as conn:
            # Overall statistics
            if game_type:
                cursor = conn.execute('''
                    SELECT
                        COUNT(*) as total_hands,
                        AVG(CASE WHEN predicted_action = actual_outcome THEN 1.0 ELSE 0.0 END) as accuracy,
                        SUM(actual_profit_loss) as total_profit,
                        AVG(actual_profit_loss) as avg_profit,
                        AVG(ABS(expected_value - actual_profit_loss)) as calibration_error
                    FROM hand_history
                    WHERE game_type = ?
                ''', (game_type,))
            else:
                cursor = conn.execute('''
                    SELECT
                        COUNT(*) as total_hands,
                        AVG(CASE WHEN predicted_action = actual_outcome THEN 1.0 ELSE 0.0 END) as accuracy,
                        SUM(actual_profit_loss) as total_profit,
                        AVG(actual_profit_loss) as avg_profit,
                        AVG(ABS(expected_value - actual_profit_loss)) as calibration_error
                    FROM hand_history
                ''')

            row = cursor.fetchone()
            if row:
                performance['total_hands'] = row['total_hands'] or 0
                performance['accuracy'] = row['accuracy'] or 0.0
                performance['total_profit'] = row['total_profit'] or 0.0
                performance['avg_profit_per_hand'] = row['avg_profit'] or 0.0
                performance['calibration_error'] = row['calibration_error'] or 0.0

            # Position-based statistics for poker
            if game_type == 'poker' or game_type is None:
                cursor = conn.execute('''
                    SELECT
                        phd.player_position,
                        COUNT(*) as hands,
                        AVG(CASE WHEN hh.predicted_action = hh.actual_outcome THEN 1.0 ELSE 0.0 END) as accuracy
                    FROM hand_history hh
                    JOIN poker_hand_details phd ON hh.id = phd.hand_history_id
                    GROUP BY phd.player_position
                ''')

                performance['accuracy_by_position'] = {
                    row['player_position']: {
                        'hands': row['hands'],
                        'accuracy': row['accuracy']
                    } for row in cursor.fetchall()
                }

            # Score-based statistics for blackjack
            if game_type == 'blackjack' or game_type is None:
                cursor = conn.execute('''
                    SELECT
                        CASE
                            WHEN bhd.player_score <= 11 THEN 'low'
                            WHEN bhd.player_score <= 16 THEN 'medium'
                            WHEN bhd.player_score <= 20 THEN 'high'
                            ELSE 'blackjack'
                        END as score_range,
                        COUNT(*) as hands,
                        AVG(CASE WHEN hh.predicted_action = hh.actual_outcome THEN 1.0 ELSE 0.0 END) as accuracy
                    FROM hand_history hh
                    JOIN blackjack_hand_details bhd ON hh.id = bhd.hand_history_id
                    GROUP BY score_range
                ''')

                performance['accuracy_by_score'] = {
                    row['score_range']: {
                        'hands': row['hands'],
                        'accuracy': row['accuracy']
                    } for row in cursor.fetchall()
                }

        return performance

    def insert_opponent_stats(self, session_id: str, opponent_id: str, game_type: str,
                            hands_played: int, vpip: float, pfr: float, af: float,
                            total_profit_loss: float) -> None:
        """Insert or update opponent statistics"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO opponent_statistics
                (session_id, opponent_id, game_type, hands_played, vpip, pfr, af, total_profit_loss, last_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, opponent_id, game_type, hands_played, vpip, pfr, af,
                  total_profit_loss, time.time()))

    def get_opponent_stats(self, session_id: str = None, opponent_id: str = None) -> List[Dict[str, Any]]:
        """Get opponent statistics with optional filtering"""
        opponents = []

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            query = 'SELECT * FROM opponent_statistics WHERE 1=1'
            params = []

            if session_id:
                query += ' AND session_id = ?'
                params.append(session_id)

            if opponent_id:
                query += ' AND opponent_id = ?'
                params.append(opponent_id)

            query += ' ORDER BY last_seen DESC'

            cursor = conn.execute(query, params)

            for row in cursor:
                opponents.append({
                    'opponent_id': row['opponent_id'],
                    'session_id': row['session_id'],
                    'game_type': row['game_type'],
                    'hands_played': row['hands_played'],
                    'vpip': row['vpip'],
                    'pfr': row['pfr'],
                    'af': row['af'],
                    'total_profit_loss': row['total_profit_loss'],
                    'last_seen': row['last_seen']
                })

        return opponents

    def update_opponent_stats_from_hands(self, session_id: str) -> None:
        """Update opponent statistics based on hand history"""
        # This would analyze hand history to calculate VPIP, PFR, etc.
        # For now, it's a placeholder for the sophisticated opponent modeling
        pass


class OpponentModel:
    """Advanced opponent modeling system with dynamic weight tables"""

    def __init__(self, opponent_id: str, game_type: str = 'poker'):
        self.opponent_id = opponent_id
        self.game_type = game_type
        self.last_updated = time.time()

        # Dynamic weight table for poker hands (1326 possible starting hands)
        self.hand_weights = self._initialize_hand_weights()

        # Behavioral statistics
        self.statistics = {
            'hands_played': 0,
            'vpip': 0.0,  # Voluntarily Put money In Pot
            'pfr': 0.0,   # Pre-Flop Raise
            'af': 0.0,    # Aggression Factor
            'cb_fold': 0.0,  # Continuation bet fold %
            'fold_to_cb': 0.0,  # Fold to continuation bet %
            '3bet': 0.0,  # 3-bet percentage
            'fold_to_3bet': 0.0,  # Fold to 3-bet %
            'steal': 0.0,  # Steal percentage from button/cutoff
            'fold_to_steal': 0.0,  # Fold to steal %
            'positional_awareness': 0.0,  # Position-based play quality
            'bluff_frequency': 0.0,  # Bluff frequency estimate
            'value_bet_frequency': 0.0,  # Value bet frequency estimate
            'tilt_factor': 0.0  # Tilt susceptibility (affects decision quality)
        }

        # Action history for pattern analysis
        self.action_history = []

        # Range analysis cache
        self.range_cache = {}

    def _initialize_hand_weights(self) -> Dict[str, float]:
        """Initialize hand weights (1326 possible starting hands)"""
        # Standard poker hand rankings for weight initialization
        hand_tiers = {
            'AA': 1.0, 'KK': 0.95, 'QQ': 0.90, 'JJ': 0.85, 'TT': 0.80,
            '99': 0.75, '88': 0.70, '77': 0.65, '66': 0.60, '55': 0.55,
            '44': 0.50, '33': 0.45, '22': 0.40,
            'AKs': 0.85, 'AQs': 0.75, 'AJs': 0.70, 'ATs': 0.65, 'A9s': 0.60,
            'A8s': 0.55, 'A7s': 0.50, 'A6s': 0.45, 'A5s': 0.40, 'A4s': 0.35,
            'A3s': 0.30, 'A2s': 0.25,
            'AKo': 0.80, 'AQo': 0.70, 'AJo': 0.65, 'ATo': 0.60, 'A9o': 0.55,
            'KQs': 0.70, 'KJs': 0.65, 'KTs': 0.60, 'K9s': 0.55, 'K8s': 0.50,
            'K7s': 0.45, 'K6s': 0.40, 'K5s': 0.35, 'K4s': 0.30, 'K3s': 0.25,
            'K2s': 0.20,
            'QJs': 0.60, 'QTs': 0.55, 'Q9s': 0.50, 'Q8s': 0.45, 'Q7s': 0.40,
            'Q6s': 0.35, 'Q5s': 0.30, 'Q4s': 0.25, 'Q3s': 0.20, 'Q2s': 0.15,
            'JTs': 0.55, 'J9s': 0.50, 'J8s': 0.45, 'J7s': 0.40, 'J6s': 0.35,
            'J5s': 0.30, 'J4s': 0.25, 'J3s': 0.20, 'J2s': 0.15,
            'T9s': 0.45, 'T8s': 0.40, 'T7s': 0.35, 'T6s': 0.30, 'T5s': 0.25,
            'T4s': 0.20, 'T3s': 0.15, 'T2s': 0.10,
            '98s': 0.35, '97s': 0.30, '96s': 0.25, '95s': 0.20, '94s': 0.15,
            '93s': 0.10, '92s': 0.05,
            '87s': 0.30, '86s': 0.25, '85s': 0.20, '84s': 0.15, '83s': 0.10,
            '82s': 0.05,
            '76s': 0.25, '75s': 0.20, '74s': 0.15, '73s': 0.10, '72s': 0.05,
            '65s': 0.20, '64s': 0.15, '63s': 0.10, '62s': 0.05,
            '54s': 0.15, '53s': 0.10, '52s': 0.05,
            '43s': 0.10, '42s': 0.05,
            '32s': 0.05
        }

        # Initialize all 1326 hands with appropriate weights
        weights = {}
        for rank1 in 'AKQJT98765432':
            for rank2 in 'AKQJT98765432':
                if rank1 == rank2:
                    # Pocket pairs
                    hand = f"{rank1}{rank2}"
                    base_rank = f"{rank1}{rank1}"
                    weights[hand] = hand_tiers.get(base_rank, 0.1)
                else:
                    # Suited and offsuit
                    for suited in ['s', 'o']:
                        hand = f"{rank1}{rank2}{suited}"
                        base_hand = f"{max(rank1, rank2)}{min(rank1, rank2)}{suited}"
                        weights[hand] = hand_tiers.get(base_hand, 0.05)

        return weights

    def update_with_action(self, game_state: Any, action: str, street: str) -> None:
        """Update model based on observed opponent action"""
        self.last_updated = time.time()
        self.statistics['hands_played'] += 1

        # Update statistics based on action and street
        self._update_statistics(game_state, action, street)

        # Update hand weights based on action
        if hasattr(game_state, 'community_cards') and game_state.community_cards:
            self._update_hand_weights(game_state, action, street)

        # Store action for pattern analysis
        action_record = {
            'timestamp': time.time(),
            'street': street,
            'action': action,
            'game_state': self._serialize_game_state(game_state)
        }
        self.action_history.append(action_record)

        # Keep only recent history for performance
        if len(self.action_history) > 200:
            self.action_history = self.action_history[-200:]

    def _update_statistics(self, game_state: Any, action: str, street: str) -> None:
        """Update behavioral statistics"""
        # VPIP calculation (any voluntary action pre-flop)
        if street == 'preflop':
            if action in ['call', 'raise', 'bet']:
                # Increment VPIP (simplified - would need more complex logic)
                self.statistics['vpip'] = min(1.0, self.statistics['vpip'] + 0.01)

        # PFR calculation (pre-flop raise)
        if street == 'preflop' and action == 'raise':
            self.statistics['pfr'] = min(1.0, self.statistics['pfr'] + 0.01)

        # Aggression factor (raises + bets vs calls + checks)
        if action in ['raise', 'bet']:
            self.statistics['af'] = min(3.0, self.statistics['af'] + 0.1)

        # Positional awareness (simplified)
        if hasattr(game_state, 'player_position'):
            position_value = self._position_to_value(game_state.player_position)
            if position_value > 0.5 and action in ['raise', 'bet']:
                self.statistics['positional_awareness'] = min(1.0, self.statistics['positional_awareness'] + 0.01)

    def _position_to_value(self, position: str) -> float:
        """Convert position to numeric value for analysis"""
        position_values = {
            'BTN': 1.0, 'CO': 0.9, 'HJ': 0.7, 'LJ': 0.6, 'MP': 0.5,
            'UTG+1': 0.4, 'UTG': 0.3, 'SB': 0.2, 'BB': 0.1
        }
        return position_values.get(position, 0.5)

    def _update_hand_weights(self, game_state: Any, action: str, street: str) -> None:
        """Update hand weights based on observed action"""
        if not hasattr(game_state, 'player_cards') or not game_state.player_cards:
            return

        # Convert cards to hand notation
        if len(game_state.player_cards) >= 2:
            card1, card2 = game_state.player_cards[:2]
            hand_notation = self._cards_to_notation(card1, card2)

            # Adjust weight based on action
            if action in ['raise', 'bet']:
                # Aggressive action - increase weight for this hand
                self.hand_weights[hand_notation] = min(1.0, self.hand_weights[hand_notation] + 0.05)
            elif action == 'fold':
                # Fold - decrease weight for this hand
                self.hand_weights[hand_notation] = max(0.0, self.hand_weights[hand_notation] - 0.02)

    def _cards_to_notation(self, card1: Any, card2: Any) -> str:
        """Convert two cards to standard poker notation"""
        rank1 = card1.rank if hasattr(card1, 'rank') else str(card1)
        rank2 = card2.rank if hasattr(card2, 'rank') else str(card2)

        # Determine suitedness
        suit1 = card1.suit if hasattr(card1, 'suit') else 'h'
        suit2 = card2.suit if hasattr(card2, 'suit') else 'd'
        suited = 's' if suit1 == suit2 else 'o'

        # Order by rank
        ranks = sorted([rank1, rank2], key=lambda x: 'AKQJT98765432'.index(x))
        return f"{ranks[1]}{ranks[0]}{suited}"

    def _serialize_game_state(self, game_state: Any) -> Dict[str, Any]:
        """Serialize game state for storage"""
        return {
            'pot_size': getattr(game_state, 'pot_size', 0),
            'player_position': getattr(game_state, 'player_position', 'unknown'),
            'community_cards': len(getattr(game_state, 'community_cards', [])),
            'current_bet': getattr(game_state, 'current_bet', 0)
        }

    def get_range_for_situation(self, situation: str, position: str = None) -> Dict[str, float]:
        """Get opponent's likely range for a specific situation"""
        if situation in self.range_cache:
            return self.range_cache[situation]

        # Generate range based on statistics and situation
        base_range = self._get_base_range(situation)

        # Adjust for opponent tendencies
        adjusted_range = self._adjust_range_for_tendencies(base_range, position)

        # Cache result
        self.range_cache[situation] = adjusted_range
        return adjusted_range

    def _get_base_range(self, situation: str) -> Dict[str, float]:
        """Get base range for situation"""
        if situation == 'open_raise':
            # Standard opening range
            return {hand: weight for hand, weight in self.hand_weights.items() if weight > 0.3}
        elif situation == 'call_raise':
            # Calling range (tighter)
            return {hand: weight for hand, weight in self.hand_weights.items() if weight > 0.5}
        elif situation == '3bet':
            # 3-betting range (very tight)
            return {hand: weight for hand, weight in self.hand_weights.items() if weight > 0.8}
        else:
            # Default range
            return {hand: weight for hand, weight in self.hand_weights.items() if weight > 0.2}

    def _adjust_range_for_tendencies(self, base_range: Dict[str, float], position: str = None) -> Dict[str, float]:
        """Adjust range based on opponent tendencies"""
        adjusted_range = base_range.copy()

        # Adjust for VPIP
        if self.statistics['vpip'] > 0.3:  # Loose player
            # Widen ranges
            for hand in adjusted_range:
                adjusted_range[hand] = min(1.0, adjusted_range[hand] * 1.2)
        elif self.statistics['vpip'] < 0.15:  # Tight player
            # Tighten ranges
            for hand in adjusted_range:
                adjusted_range[hand] = max(0.0, adjusted_range[hand] * 0.8)

        # Adjust for position if available
        if position:
            position_factor = self._position_to_value(position)
            for hand in adjusted_range:
                adjusted_range[hand] *= position_factor

        return adjusted_range

    def get_action_probability(self, situation: str, possible_actions: List[str]) -> Dict[str, float]:
        """Get probability of different actions in a situation"""
        probabilities = {}

        for action in possible_actions:
            base_prob = 0.2  # Default probability

            # Adjust based on aggression factor
            if action in ['raise', 'bet']:
                base_prob *= (1 + self.statistics['af'])
            elif action == 'call':
                base_prob *= (1 - self.statistics['af'] * 0.5)
            elif action == 'fold':
                base_prob *= (1 - self.statistics['vpip'])

            probabilities[action] = min(1.0, max(0.0, base_prob))

        # Normalize probabilities
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}

        return probabilities


class OpponentModelManager:
    """Manager for multiple opponent models"""

    def __init__(self):
        self.models: Dict[str, OpponentModel] = {}
        self.session_id: str = None

    def set_session(self, session_id: str) -> None:
        """Set current session for opponent tracking"""
        self.session_id = session_id

    def get_or_create_model(self, opponent_id: str, game_type: str = 'poker') -> OpponentModel:
        """Get existing model or create new one"""
        model_key = f"{opponent_id}_{game_type}"

        if model_key not in self.models:
            self.models[model_key] = OpponentModel(opponent_id, game_type)

        return self.models[model_key]

    def update_opponent_action(self, opponent_id: str, game_state: Any, action: str, street: str) -> None:
        """Update opponent model with new action"""
        model = self.get_or_create_model(opponent_id, getattr(game_state, 'game_type', 'poker'))
        model.update_with_action(game_state, action, street)

    def get_opponent_range(self, opponent_id: str, situation: str, position: str = None) -> Dict[str, float]:
        """Get opponent's likely hand range"""
        model = self.get_or_create_model(opponent_id)
        return model.get_range_for_situation(situation, position)

    def save_models_to_database(self, db_manager: DatabaseManager) -> None:
        """Save opponent models to database"""
        for model in self.models.values():
            # Calculate aggregate statistics
            vpip = model.statistics['vpip']
            pfr = model.statistics['pfr']
            af = model.statistics['af']

            # Calculate total profit/loss from action history
            total_pl = sum(record.get('game_state', {}).get('pot_size', 0)
                          for record in model.action_history[-50:])  # Last 50 actions

            db_manager.insert_opponent_stats(
                self.session_id, model.opponent_id, model.game_type,
                model.statistics['hands_played'], vpip, pfr, af, total_pl
            )


class UnifiedHistoricalTracker:
    """Unified system for tracking poker and blackjack performance"""

    def __init__(self):
        self.history_file = 'ai_agent_history.json'
        self.sessions_file = 'gaming_sessions.json'
        self.db_manager = DatabaseManager()
        self.hand_history: List[HandResult] = []
        self.active_sessions: Dict[str, GameSession] = {}
        self.model_performance = {
            'poker': {
                'total_hands': 0,
                'correct_predictions': 0,
                'accuracy_by_position': {},
                'profit_loss_tracking': [],
                'expected_vs_actual': []
            },
            'blackjack': {
                'total_hands': 0,
                'correct_predictions': 0,
                'accuracy_by_score': {},
                'profit_loss_tracking': [],
                'true_count_effectiveness': []
            }
        }
        self._load_all_history()

    def _load_all_history(self) -> None:
        """Load all historical data"""
        self._load_hand_history()
        self._load_sessions()
        self._calculate_model_performance()

    def _load_hand_history(self) -> None:
        """Load hand history from file"""
        try:
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.hand_history = [HandResult(**hand_data) for hand_data in data.get('hand_history', [])]
        except (FileNotFoundError, json.JSONDecodeError):
            self.hand_history = []

    def _load_sessions(self) -> None:
        """Load session data"""
        try:
            with open(self.sessions_file, 'r') as f:
                data = json.load(f)
                self.active_sessions = {k: GameSession(**v) for k, v in data.get('sessions', {}).items()}
        except (FileNotFoundError, json.JSONDecodeError):
            self.active_sessions = {}

    def _save_hand_history(self) -> None:
        """Save hand history to file"""
        try:
            data = {
                'hand_history': [self._hand_result_to_dict(hand) for hand in self.hand_history],
                'last_updated': time.time()
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save hand history: {e}")

    def _save_sessions(self) -> None:
        """Save session data"""
        try:
            data = {
                'sessions': {k: self._session_to_dict(v) for k, v in self.active_sessions.items()},
                'last_updated': time.time()
            }
            with open(self.sessions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")

    def _hand_result_to_dict(self, hand: HandResult) -> Dict[str, Any]:
        """Convert HandResult to dictionary for JSON serialization"""
        return {
            'game_type': hand.game_type,
            'timestamp': hand.timestamp,
            'predicted_action': hand.predicted_action,
            'actual_outcome': hand.actual_outcome,
            'expected_value': hand.expected_value,
            'actual_profit_loss': hand.actual_profit_loss,
            'game_state_summary': hand.game_state_summary,
            'confidence_score': hand.confidence_score,
            'model_version': hand.model_version
        }

    def _session_to_dict(self, session: GameSession) -> Dict[str, Any]:
        """Convert GameSession to dictionary for JSON serialization"""
        return {
            'session_id': session.session_id,
            'game_type': session.game_type,
            'start_time': session.start_time,
            'end_time': session.end_time,
            'hands_played': session.hands_played,
            'total_profit_loss': session.total_profit_loss,
            'location': session.location
        }

    def _calculate_model_performance(self) -> None:
        """Recalculate model performance from database"""
        # Get fresh performance data from database
        poker_performance = self.db_manager.get_model_performance('poker')
        blackjack_performance = self.db_manager.get_model_performance('blackjack')

        # Update in-memory performance for quick access
        self.model_performance = {
            'poker': {
                'total_hands': poker_performance.get('total_hands', 0),
                'correct_predictions': int(poker_performance.get('accuracy', 0) * poker_performance.get('total_hands', 0)),
                'accuracy_by_position': poker_performance.get('accuracy_by_position', {}),
                'profit_loss_tracking': [],  # Would need separate query for this
                'expected_vs_actual': [poker_performance.get('calibration_error', 0)]
            },
            'blackjack': {
                'total_hands': blackjack_performance.get('total_hands', 0),
                'correct_predictions': int(blackjack_performance.get('accuracy', 0) * blackjack_performance.get('total_hands', 0)),
                'accuracy_by_score': blackjack_performance.get('accuracy_by_score', {}),
                'profit_loss_tracking': [],  # Would need separate query for this
                'true_count_effectiveness': []  # Would need separate query for this
            }
        }

        # Update active sessions from database
        with sqlite3.connect(self.db_manager.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute('SELECT * FROM gaming_sessions WHERE end_time IS NULL')
            self.active_sessions = {}

            for row in cursor:
                session = GameSession(
                    session_id=row['session_id'],
                    game_type=row['game_type'],
                    start_time=row['start_time'],
                    end_time=row['end_time'],
                    hands_played=row['hands_played'],
                    total_profit_loss=row['total_profit_loss'],
                    location=row['location']
                )
                self.active_sessions[session.session_id] = session

    def _track_poker_performance(self, hand: HandResult) -> None:
        """Track poker-specific performance metrics"""
        position = hand.game_state_summary.get('player_position', 'unknown')
        if position not in self.model_performance['poker']['accuracy_by_position']:
            self.model_performance['poker']['accuracy_by_position'][position] = {'correct': 0, 'total': 0}

        self.model_performance['poker']['accuracy_by_position'][position]['total'] += 1
        if hand.predicted_action == hand.actual_outcome:
            self.model_performance['poker']['accuracy_by_position'][position]['correct'] += 1

    def _track_blackjack_performance(self, hand: HandResult) -> None:
        """Track blackjack-specific performance metrics"""
        player_score = hand.game_state_summary.get('player_score', 0)
        score_range = self._categorize_blackjack_score(player_score)

        if score_range not in self.model_performance['blackjack']['accuracy_by_score']:
            self.model_performance['blackjack']['accuracy_by_score'][score_range] = {'correct': 0, 'total': 0}

        self.model_performance['blackjack']['accuracy_by_score'][score_range]['total'] += 1
        if hand.predicted_action == hand.actual_outcome:
            self.model_performance['blackjack']['accuracy_by_score'][score_range]['correct'] += 1

        # Track true count effectiveness
        true_count = hand.game_state_summary.get('true_count', 0)
        if true_count != 0:
            self.model_performance['blackjack']['true_count_effectiveness'].append(abs(true_count))

    def _categorize_blackjack_score(self, score: int) -> str:
        """Categorize blackjack hand score"""
        if score <= 11:
            return 'low'
        elif score <= 16:
            return 'medium'
        elif score <= 20:
            return 'high'
        else:
            return 'blackjack'

    def start_session(self, game_type: str, location: str = None) -> str:
        """Start a new gaming session"""
        session_id = f"{game_type}_{int(time.time())}_{len(self.active_sessions)}"
        session = GameSession(
            session_id=session_id,
            game_type=game_type,
            start_time=time.time(),
            location=location
        )
        self.active_sessions[session_id] = session
        self._save_sessions()
        logger.info(f"Started {game_type} session: {session_id}")
        return session_id

    def end_session(self, session_id: str) -> None:
        """End an active gaming session"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.end_time = time.time()
            logger.info(f"Ended session {session_id}: {session.hands_played} hands, "
                       f"P&L: ${session.total_profit_loss:.2f}")
            self._save_sessions()

    def record_game_result(self, game_type: str, game_state: Any, predicted_action: str,
                           actual_outcome: str, expected_value: float, actual_profit_loss: float,
                           session_id: str = None) -> None:
        """Record result of a game hand"""

        # Create game state summary
        if game_type == 'poker' and isinstance(game_state, PokerGameState):
            game_state_summary = {
                'player_position': game_state.player_position,
                'game_phase': game_state.game_phase,
                'pot_size': game_state.pot_size,
                'player_stack': game_state.player_stack,
                'hand_strength_category': 'unknown'  # Would be calculated
            }
        elif game_type == 'blackjack' and isinstance(game_state, BlackjackGameState):
            game_state_summary = {
                'player_score': game_state.player_score,
                'dealer_upcard': game_state.dealer_upcard.value if game_state.dealer_upcard else 0,
                'true_count': game_state.true_count,
                'current_bet': game_state.current_bet
            }
        else:
            game_state_summary = {}

        # Create hand result for in-memory operations
        hand_result = HandResult(
            game_type=game_type,
            timestamp=time.time(),
            predicted_action=predicted_action,
            actual_outcome=actual_outcome,
            expected_value=expected_value,
            actual_profit_loss=actual_profit_loss,
            game_state_summary=game_state_summary,
            confidence_score=getattr(game_state, 'confidence', 0.5)
        )

        # Store in database for persistent storage
        hand_id = self.db_manager.insert_hand_result(hand_result)

        # Update session in database if active
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.hands_played += 1
            session.total_profit_loss += actual_profit_loss
            self.db_manager.insert_session(session)

        # Add to in-memory history for quick access (keep recent hands only)
        self.hand_history.append(hand_result)

        # Recalculate performance metrics from database
        self._calculate_model_performance()

        # Keep only recent history in memory (last 100 hands for performance)
        if len(self.hand_history) > 100:
            self.hand_history = self.hand_history[-100:]

        # Save JSON backups periodically (every 50 hands)
        if len(self.hand_history) % 50 == 0:
            self._save_hand_history()
            self._save_sessions()

    def get_overall_accuracy(self) -> float:
        """Get overall model accuracy across both games"""
        total_hands = sum(game_data['total_hands'] for game_data in self.model_performance.values())
        total_correct = sum(game_data['correct_predictions'] for game_data in self.model_performance.values())

        return total_correct / total_hands if total_hands > 0 else 0.0

    def get_game_accuracy(self, game_type: str) -> float:
        """Get accuracy for specific game"""
        if game_type not in self.model_performance:
            return 0.0

        game_data = self.model_performance[game_type]
        return game_data['correct_predictions'] / game_data['total_hands'] if game_data['total_hands'] > 0 else 0.0

    def get_profitability_metrics(self, game_type: str = None) -> Dict[str, float]:
        """Get profitability metrics for specific game or overall"""
        if game_type:
            if game_type not in self.model_performance:
                return {'total_profit': 0.0, 'avg_profit_per_hand': 0.0, 'win_rate': 0.0}

            profits = self.model_performance[game_type]['profit_loss_tracking']
            if not profits:
                return {'total_profit': 0.0, 'avg_profit_per_hand': 0.0, 'win_rate': 0.0}

            total_profit = sum(profits)
            winning_hands = sum(1 for p in profits if p > 0)

            return {
                'total_profit': total_profit,
                'avg_profit_per_hand': total_profit / len(profits),
                'win_rate': winning_hands / len(profits)
            }
        else:
            # Overall metrics
            all_profits = []
            for game_data in self.model_performance.values():
                all_profits.extend(game_data['profit_loss_tracking'])

            if not all_profits:
                return {'total_profit': 0.0, 'avg_profit_per_hand': 0.0, 'win_rate': 0.0}

            total_profit = sum(all_profits)
            winning_hands = sum(1 for p in all_profits if p > 0)

            return {
                'total_profit': total_profit,
                'avg_profit_per_hand': total_profit / len(all_profits),
                'win_rate': winning_hands / len(all_profits)
            }

    def get_model_calibration_error(self, game_type: str = None) -> float:
        """Get average difference between expected and actual results"""
        if game_type:
            if game_type not in self.model_performance:
                return 0.0

            errors = self.model_performance[game_type]['expected_vs_actual']
            return sum(errors) / len(errors) if errors else 0.0
        else:
            # Overall calibration error
            all_errors = []
            for game_data in self.model_performance.values():
                all_errors.extend(game_data['expected_vs_actual'])

            return sum(all_errors) / len(all_errors) if all_errors else 0.0

    def get_model_improvements(self, game_type: str = None) -> Dict[str, Any]:
        """Analyze performance and suggest improvements"""
        improvements = {
            'accuracy_trends': {},
            'profitable_situations': [],
            'problematic_areas': [],
            'calibration_issues': []
        }

        if game_type:
            game_data = self.model_performance.get(game_type, {})
        else:
            # Overall analysis
            game_data = {
                'accuracy_by_position': {},
                'accuracy_by_score': {}
            }
            for game in self.model_performance.values():
                for pos, stats in game.get('accuracy_by_position', {}).items():
                    if pos not in game_data['accuracy_by_position']:
                        game_data['accuracy_by_position'][pos] = {'correct': 0, 'total': 0}
                    game_data['accuracy_by_position'][pos]['correct'] += stats['correct']
                    game_data['accuracy_by_position'][pos]['total'] += stats['total']

                for score, stats in game.get('accuracy_by_score', {}).items():
                    if score not in game_data['accuracy_by_score']:
                        game_data['accuracy_by_score'][score] = {'correct': 0, 'total': 0}
                    game_data['accuracy_by_score'][score]['correct'] += stats['correct']
                    game_data['accuracy_by_score'][score]['total'] += stats['total']

        # Analyze accuracy by categories
        if 'accuracy_by_position' in game_data:
            for position, stats in game_data['accuracy_by_position'].items():
                if stats['total'] > 20:  # Minimum sample size
                    accuracy = stats['correct'] / stats['total']
                    improvements['accuracy_trends'][position] = accuracy
                    if accuracy > 0.7:
                        improvements['profitable_situations'].append(f"Good accuracy in {position}")
                    elif accuracy < 0.5:
                        improvements['problematic_areas'].append(f"Poor accuracy in {position}")

        if 'accuracy_by_score' in game_data:
            for score_range, stats in game_data['accuracy_by_score'].items():
                if stats['total'] > 20:
                    accuracy = stats['correct'] / stats['total']
                    if accuracy < 0.6:
                        improvements['problematic_areas'].append(f"Poor accuracy for {score_range} scores")

        # Check calibration
        calibration_error = self.get_model_calibration_error(game_type)
        if calibration_error > 1.0:
            improvements['calibration_issues'].append(f"High calibration error: ${calibration_error:.2f} per hand")

        return improvements

    def generate_model_adjustments(self) -> Dict[str, Any]:
        """Generate model adjustments based on historical performance"""
        adjustments = {
            'poker_adjustments': {},
            'blackjack_adjustments': {},
            'confidence_calibration': {},
            'strategy_improvements': []
        }

        # Analyze poker performance by position
        poker_data = self.model_performance.get('poker', {})
        if 'accuracy_by_position' in poker_data:
            for position, stats in poker_data['accuracy_by_position'].items():
                if stats['total'] > 50:  # Minimum sample size
                    accuracy = stats['correct'] / stats['total']
                    if accuracy < 0.6:
                        # Suggest more conservative play in this position
                        adjustments['poker_adjustments'][position] = {
                            'suggested_change': 'more_conservative',
                            'confidence_threshold_increase': 0.1,
                            'reason': f'Low accuracy ({accuracy:.1%}) in {position} position'
                        }

        # Analyze blackjack performance by score
        blackjack_data = self.model_performance.get('blackjack', {})
        if 'accuracy_by_score' in blackjack_data:
            for score_range, stats in blackjack_data['accuracy_by_score'].items():
                if stats['total'] > 30:
                    accuracy = stats['correct'] / stats['total']
                    if accuracy < 0.65:
                        adjustments['blackjack_adjustments'][score_range] = {
                            'suggested_change': 'more_conservative',
                            'ev_threshold_increase': 0.05,
                            'reason': f'Low accuracy ({accuracy:.1%}) for {score_range} scores'
                        }

        # Confidence calibration based on historical accuracy
        overall_accuracy = self.get_overall_accuracy()
        if overall_accuracy > 0:
            if overall_accuracy > 0.8:
                adjustments['confidence_calibration'] = {
                    'multiplier': 1.1,  # Increase confidence
                    'reason': 'High historical accuracy'
                }
            elif overall_accuracy < 0.6:
                adjustments['confidence_calibration'] = {
                    'multiplier': 0.9,  # Decrease confidence
                    'reason': 'Low historical accuracy'
                }

        # Strategy improvements based on profitability analysis
        profitability = self.get_profitability_metrics()
        if profitability['win_rate'] < 0.4:
            adjustments['strategy_improvements'].append({
                'area': 'general',
                'suggestion': 'Consider more conservative betting',
                'reason': f'Low win rate: {profitability["win_rate"]:.1%}'
            })

        return adjustments

    def update_model_parameters(self, game_type: str, adjustments: Dict[str, Any]) -> None:
        """Update model parameters based on improvements"""
        try:
            # This would update the actual model parameters
            # For now, we'll just log the adjustments
            logger.info(f"Model adjustments for {game_type}: {adjustments}")

            # In a real implementation, this would modify:
            # - Confidence thresholds
            # - Strategy tables
            # - Risk parameters
            # - Opponent modeling parameters

        except Exception as e:
            logger.error(f"Failed to update model parameters: {e}")

    def predict_future_performance(self, game_type: str) -> Dict[str, float]:
        """Predict future performance based on historical trends"""
        predictions = {
            'expected_accuracy': 0.0,
            'expected_profit_per_hand': 0.0,
            'confidence_interval': 0.0,
            'trend_direction': 'stable'
        }

        try:
            # Get recent performance (last 100 hands)
            recent_hands = self.hand_history[-100:] if len(self.hand_history) > 100 else self.hand_history

            if game_type == 'poker':
                game_hands = [h for h in recent_hands if h.game_type == 'poker']
            elif game_type == 'blackjack':
                game_hands = [h for h in recent_hands if h.game_type == 'blackjack']
            else:
                return predictions

            if not game_hands:
                return predictions

            # Calculate recent accuracy trend
            recent_correct = sum(1 for h in game_hands if h.predicted_action == h.actual_outcome)
            recent_accuracy = recent_correct / len(game_hands)

            # Calculate profit trend
            recent_profits = [h.actual_profit_loss for h in game_hands]
            avg_recent_profit = sum(recent_profits) / len(recent_profits)

            # Simple trend analysis (compare recent vs overall)
            overall_accuracy = self.get_game_accuracy(game_type)
            overall_profit = self.get_profitability_metrics(game_type)['avg_profit_per_hand']

            # Determine trend direction
            accuracy_trend = recent_accuracy - overall_accuracy
            profit_trend = avg_recent_profit - overall_profit

            if accuracy_trend > 0.05:
                predictions['trend_direction'] = 'improving'
            elif accuracy_trend < -0.05:
                predictions['trend_direction'] = 'declining'
            else:
                predictions['trend_direction'] = 'stable'

            predictions['expected_accuracy'] = recent_accuracy
            predictions['expected_profit_per_hand'] = avg_recent_profit
            predictions['confidence_interval'] = min(abs(accuracy_trend), abs(profit_trend))

        except Exception as e:
            logger.error(f"Failed to predict future performance: {e}")

        return predictions

    def export_training_data(self, filename: str = None) -> str:
        """Export data for external model training and analysis"""
        if not filename:
            filename = f'ai_agent_training_data_{int(time.time())}.json'

        try:
            # Prepare training data format
            training_data = {
                'metadata': {
                    'export_timestamp': time.time(),
                    'total_samples': len(self.hand_history),
                    'games_included': list(set(h.game_type for h in self.hand_history))
                },
                'training_samples': []
            }

            for hand in self.hand_history:
                sample = {
                    'input_features': {
                        'game_type': hand.game_type,
                        'game_state': hand.game_state_summary,
                        'predicted_action': hand.predicted_action
                    },
                    'output_label': hand.actual_outcome,
                    'reward': hand.actual_profit_loss,
                    'expected_reward': hand.expected_value,
                    'confidence': hand.confidence_score
                }
                training_data['training_samples'].append(sample)

            with open(filename, 'w') as f:
                json.dump(training_data, f, indent=2, default=str)

            logger.info(f"Training data exported to {filename} ({len(training_data['training_samples'])} samples)")
            return filename

        except Exception as e:
            logger.error(f"Failed to export training data: {e}")
            return None

    def export_data(self, filename: str = None) -> str:
        """Export historical data for analysis"""
        if not filename:
            filename = f'ai_agent_export_{int(time.time())}.json'

        try:
            export_data = {
                'export_timestamp': time.time(),
                'summary': {
                    'total_hands': len(self.hand_history),
                    'accuracy': self.get_overall_accuracy(),
                    'total_profit': self.get_profitability_metrics()['total_profit'],
                    'model_performance': self.model_performance
                },
                'hand_history': [self._hand_result_to_dict(hand) for hand in self.hand_history],
                'active_sessions': {k: self._session_to_dict(v) for k, v in self.active_sessions.items()},
                'model_improvements': self.get_model_improvements()
            }

            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Data exported to {filename}")
            return filename

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            return None


# Global tracker instance
historical_tracker = UnifiedHistoricalTracker()