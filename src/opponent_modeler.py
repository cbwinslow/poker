"""
Advanced Opponent Modeling System (Expert Analysis Implementation)
Sophisticated opponent behavior modeling using LSTM and Transformer architectures
"""
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class OpponentAction:
    """Represents a single opponent action for modeling"""
    timestamp: float
    action_type: str  # 'fold', 'call', 'raise', 'check'
    bet_size: float   # Relative bet size (0-1 as fraction of pot)
    position: str     # Player position
    game_phase: str   # preflop, flop, turn, river
    pot_size: float   # Current pot size
    stack_size: float # Opponent's stack size

@dataclass
class OpponentProfile:
    """Comprehensive opponent behavioral profile"""
    opponent_id: str
    total_hands: int
    vpip: float           # Voluntarily Put money In Pot
    pfr: float            # Pre-Flop Raise
    af: float             # Aggression Factor
    three_bet: float      # 3-bet frequency
    fold_to_three_bet: float
    cbet_flop: float      # Continuation bet on flop
    fold_to_cbet: float   # Fold to continuation bet
    style_category: str   # TAG, LAG, NIT, MANIAC, etc.
    adaptability_score: float  # How much style changes over time

class AdvancedOpponentModeler:
    """Advanced opponent modeling using LSTM and Transformer approaches (expert analysis)"""

    def __init__(self):
        # Sequential modeling (LSTM approach from expert analysis)
        self.game_module_weights = {}      # Per-hand tactical modeling
        self.opponent_module_weights = {}  # Long-term strategic modeling

        # Transformer-based modeling (modern approach)
        self.transformer_beliefs = {}      # Attention-based belief states

        # Opponent profiles and history
        self.opponent_profiles = {}        # Comprehensive behavioral profiles
        self.action_histories = {}         # Complete action sequences

        # Performance tracking
        self.modeling_accuracy = {}        # Track prediction accuracy per opponent

        # Feature engineering for modeling
        self.feature_scales = {
            'bet_size': (0, 10),          # Relative to pot size
            'pot_size': (0, 100),         # In big blinds
            'stack_size': (0, 200),       # In big blinds
            'position_value': (0, 1),     # Positional advantage
            'game_phase': (0, 3)          # preflop=0, flop=1, turn=2, river=3
        }

    def update_opponent_model(self, opponent_id: str, action: OpponentAction) -> None:
        """Update opponent model with new action (expert analysis)"""

        # Initialize opponent tracking if new
        if opponent_id not in self.opponent_profiles:
            self._initialize_opponent_profile(opponent_id)

        if opponent_id not in self.action_histories:
            self.action_histories[opponent_id] = []

        # Store action for sequence modeling
        self.action_histories[opponent_id].append(action)

        # Keep only recent actions (last 1000)
        if len(self.action_histories[opponent_id]) > 1000:
            self.action_histories[opponent_id] = self.action_histories[opponent_id][-1000:]

        # Update statistical profile
        self._update_statistical_profile(opponent_id, action)

        # Update sequential models (LSTM approach)
        self._update_sequential_models(opponent_id, action)

        # Update transformer beliefs (modern approach)
        self._update_transformer_beliefs(opponent_id, action)

    def _initialize_opponent_profile(self, opponent_id: str) -> None:
        """Initialize new opponent profile"""
        self.opponent_profiles[opponent_id] = OpponentProfile(
            opponent_id=opponent_id,
            total_hands=0,
            vpip=0.0,
            pfr=0.0,
            af=0.0,
            three_bet=0.0,
            fold_to_three_bet=0.0,
            cbet_flop=0.0,
            fold_to_cbet=0.0,
            style_category='unknown',
            adaptability_score=0.0
        )

        self.action_histories[opponent_id] = []
        self.modeling_accuracy[opponent_id] = []

    def _update_statistical_profile(self, opponent_id: str, action: OpponentAction) -> None:
        """Update statistical behavioral profile"""
        profile = self.opponent_profiles[opponent_id]
        profile.total_hands += 1

        # Update VPIP (any voluntary action preflop)
        if action.game_phase == 'preflop' and action.action_type in ['call', 'raise']:
            profile.vpip = ((profile.vpip * (profile.total_hands - 1)) + 1) / profile.total_hands

        # Update PFR (preflop raises)
        if action.game_phase == 'preflop' and action.action_type == 'raise':
            profile.pfr = ((profile.pfr * (profile.total_hands - 1)) + 1) / profile.total_hands

        # Update aggression factor (raises / calls)
        if action.action_type in ['call', 'raise']:
            total_aggressive_actions = sum(1 for a in self.action_histories[opponent_id]
                                         if a.action_type in ['raise'])
            total_passive_actions = sum(1 for a in self.action_histories[opponent_id]
                                      if a.action_type == 'call')

            if total_passive_actions > 0:
                profile.af = total_aggressive_actions / total_passive_actions

        # Update style categorization
        profile.style_category = self._categorize_playing_style(profile)

    def _categorize_playing_style(self, profile: OpponentProfile) -> str:
        """Categorize opponent playing style based on statistics"""
        # TAG (Tight-Aggressive): Low VPIP, High PFR, High AF
        if profile.vpip < 0.15 and profile.pfr > 0.10 and profile.af > 1.5:
            return 'TAG'

        # LAG (Loose-Aggressive): High VPIP, High PFR, High AF
        elif profile.vpip > 0.25 and profile.pfr > 0.15 and profile.af > 2.0:
            return 'LAG'

        # NIT (Tight-Passive): Low VPIP, Low PFR, Low AF
        elif profile.vpip < 0.12 and profile.pfr < 0.08 and profile.af < 1.0:
            return 'NIT'

        # MANIAC (Loose-Passive): High VPIP, Low PFR, Variable AF
        elif profile.vpip > 0.30 and profile.pfr < 0.10:
            return 'MANIAC'

        # FISH (Recreational): High VPIP, Low PFR, Low AF
        elif profile.vpip > 0.20 and profile.pfr < 0.08 and profile.af < 1.2:
            return 'FISH'

        else:
            return 'unknown'

    def _update_sequential_models(self, opponent_id: str, action: OpponentAction) -> None:
        """Update LSTM-style sequential models (expert analysis)"""

        # Convert action to feature vector
        features = self._action_to_features(action)

        # Game module: Tactical, per-hand modeling
        game_state = self._update_game_module(opponent_id, features)

        # Opponent module: Strategic, long-term modeling
        opponent_state = self._update_opponent_module(opponent_id, features)

        # Store updated states for next iteration
        self.game_module_weights[opponent_id] = game_state
        self.opponent_module_weights[opponent_id] = opponent_state

    def _update_transformer_beliefs(self, opponent_id: str, action: OpponentAction) -> None:
        """Update Transformer-based belief modeling (expert analysis)"""

        # Create attention-based belief update
        action_features = self._action_to_features(action)

        if opponent_id not in self.transformer_beliefs:
            # Initialize with uniform attention
            self.transformer_beliefs[opponent_id] = {
                'belief_state': self._initialize_uniform_belief_state(),
                'attention_weights': np.ones(100) / 100,  # Uniform attention
                'memory_tokens': []
            }

        # Update belief using attention mechanism
        updated_belief = self._attention_based_belief_update(
            self.transformer_beliefs[opponent_id]['belief_state'],
            action_features
        )

        self.transformer_beliefs[opponent_id]['belief_state'] = updated_belief

        # Store in memory for future attention
        self.transformer_beliefs[opponent_id]['memory_tokens'].append(action_features)
        if len(self.transformer_beliefs[opponent_id]['memory_tokens']) > 100:
            self.transformer_beliefs[opponent_id]['memory_tokens'] = \
                self.transformer_beliefs[opponent_id]['memory_tokens'][-100:]

    def _action_to_features(self, action: OpponentAction) -> np.ndarray:
        """Convert action to feature vector for modeling"""
        features = np.zeros(10)  # 10-dimensional feature vector

        # Normalize features based on scales
        features[0] = min(1.0, action.bet_size / 10.0)  # Bet size (0-1)
        features[1] = min(1.0, action.pot_size / 100.0)  # Pot size (0-1)
        features[2] = min(1.0, action.stack_size / 200.0)  # Stack size (0-1)

        # Positional encoding
        position_encoding = {'BTN': 1.0, 'CO': 0.9, 'HJ': 0.7, 'LJ': 0.5, 'MP': 0.4,
                           'UTG': 0.2, 'SB': 0.1, 'BB': 0.0}
        features[3] = position_encoding.get(action.position, 0.0)

        # Game phase encoding
        phase_encoding = {'preflop': 0.0, 'flop': 0.33, 'turn': 0.67, 'river': 1.0}
        features[4] = phase_encoding.get(action.game_phase, 0.0)

        # Action type encoding
        action_encoding = {'fold': 0.0, 'check': 0.25, 'call': 0.5, 'raise': 0.75}
        features[5] = action_encoding.get(action.action_type, 0.0)

        return features

    def _update_game_module(self, opponent_id: str, features: np.ndarray) -> np.ndarray:
        """Update LSTM-style game module (tactical modeling)"""
        # Simplified LSTM-like state update
        if opponent_id not in self.game_module_weights:
            # Initialize with zeros (like LSTM hidden state)
            state = np.zeros(32)  # 32-dimensional state
        else:
            state = self.game_module_weights[opponent_id]

        # Simple state update (would be LSTM cell in real implementation)
        # Forget gate (simplified)
        forget_factor = 0.9
        state = state * forget_factor

        # Input gate (simplified)
        input_factor = 0.1
        state = state + (features[:32] * input_factor)  # Take first 32 features

        return state

    def _update_opponent_module(self, opponent_id: str, features: np.ndarray) -> np.ndarray:
        """Update LSTM-style opponent module (strategic modeling)"""
        # Similar to game module but with longer memory
        if opponent_id not in self.opponent_module_weights:
            state = np.zeros(64)  # Larger state for strategic modeling
        else:
            state = self.opponent_module_weights[opponent_id]

        # Longer-term memory (less forgetting)
        forget_factor = 0.95
        state = state * forget_factor

        # Strategic feature integration
        input_factor = 0.05  # Slower learning for strategic patterns
        state = state + (features * input_factor)

        return state

    def _initialize_uniform_belief_state(self) -> Dict[str, float]:
        """Initialize uniform belief distribution over possible hands"""
        # 169 possible starting hands in Texas Hold'em (13*13)
        possible_hands = []
        ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']

        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                if i >= j:  # Avoid duplicates (AA, AKs, etc.)
                    hand = f"{rank1}{rank2}{'s' if i != j else ''}"
                    possible_hands.append(hand)

        uniform_prob = 1.0 / len(possible_hands)
        return {hand: uniform_prob for hand in possible_hands[:100]}  # Limit for performance

    def _attention_based_belief_update(self, current_belief: Dict[str, float],
                                     action_features: np.ndarray) -> Dict[str, float]:
        """Update belief state using attention mechanism (Transformer approach)"""

        # Compute attention weights based on action features
        attention_scores = {}
        for hand, prior_prob in current_belief.items():
            # Simple attention score based on hand strength and action
            hand_strength = self._estimate_hand_strength_from_action(hand, action_features)
            attention_scores[hand] = prior_prob * hand_strength

        # Normalize to get updated belief distribution
        total_score = sum(attention_scores.values())
        if total_score > 0:
            updated_belief = {hand: score / total_score for hand, score in attention_scores.items()}
        else:
            updated_belief = current_belief  # No change if no information

        return updated_belief

    def _estimate_hand_strength_from_action(self, hand: str, action_features: np.ndarray) -> float:
        """Estimate hand strength based on action and context"""
        # Simplified hand strength estimation
        hand_rankings = {
            'AA': 1.0, 'KK': 0.95, 'QQ': 0.9, 'JJ': 0.85, 'TT': 0.8,
            'AK': 0.75, 'AQ': 0.7, 'AJ': 0.65, 'KQ': 0.6, 'KJ': 0.55,
            '22': 0.3, '32': 0.1, '72': 0.05
        }

        base_strength = hand_rankings.get(hand[:2], 0.5)

        # Adjust based on action features
        bet_size_factor = action_features[0]  # Larger bets suggest stronger hands
        position_factor = action_features[3]  # Better position suggests wider range

        adjusted_strength = base_strength * (1 + bet_size_factor * 0.5) * (1 + position_factor * 0.3)

        return min(1.0, adjusted_strength)

    def predict_opponent_action(self, opponent_id: str, game_context: Dict[str, Any]) -> str:
        """Predict opponent's likely next action using ensemble modeling"""

        if opponent_id not in self.opponent_profiles:
            return 'unknown'

        profile = self.opponent_profiles[opponent_id]

        # Ensemble prediction using multiple modeling approaches
        predictions = {}

        # Statistical prediction
        predictions['statistical'] = self._predict_from_statistics(profile, game_context)

        # Sequential prediction (LSTM approach)
        if opponent_id in self.action_histories:
            predictions['sequential'] = self._predict_from_sequential_model(opponent_id, game_context)

        # Transformer prediction
        if opponent_id in self.transformer_beliefs:
            predictions['transformer'] = self._predict_from_transformer_model(opponent_id, game_context)

        # Ensemble combination (weighted by confidence)
        final_prediction = self._ensemble_predictions(predictions, game_context)

        return final_prediction

    def _predict_from_statistics(self, profile: OpponentProfile, game_context: Dict[str, Any]) -> str:
        """Make prediction based on statistical profile"""
        # Simplified statistical prediction
        if game_context.get('game_phase') == 'preflop':
            if profile.pfr > 0.15:
                return 'raise'
            elif profile.vpip > 0.20:
                return 'call'
            else:
                return 'fold'
        else:
            if profile.af > 1.5:
                return 'raise'
            else:
                return 'call'

    def _predict_from_sequential_model(self, opponent_id: str, game_context: Dict[str, Any]) -> str:
        """Make prediction using LSTM-style sequential modeling"""
        # Simplified sequential prediction
        recent_actions = self.action_histories[opponent_id][-10:]  # Last 10 actions

        aggressive_count = sum(1 for action in recent_actions if action.action_type == 'raise')
        passive_count = sum(1 for action in recent_actions if action.action_type == 'call')

        if aggressive_count > passive_count:
            return 'raise'
        else:
            return 'call'

    def _predict_from_transformer_model(self, opponent_id: str, game_context: Dict[str, Any]) -> str:
        """Make prediction using Transformer belief modeling"""
        # Use belief state to inform prediction
        belief_state = self.transformer_beliefs[opponent_id]['belief_state']

        # Find most likely hands
        top_hands = sorted(belief_state.items(), key=lambda x: x[1], reverse=True)[:5]

        # Predict based on most likely hand range
        avg_strength = sum(prob for _, prob in top_hands) / len(top_hands)

        if avg_strength > 0.7:
            return 'raise'
        elif avg_strength > 0.4:
            return 'call'
        else:
            return 'fold'

    def _ensemble_predictions(self, predictions: Dict[str, str], game_context: Dict[str, Any]) -> str:
        """Combine multiple predictions using ensemble method"""

        # Count votes for each action
        action_votes = {}
        for method, prediction in predictions.items():
            action_votes[prediction] = action_votes.get(prediction, 0) + 1

        # Return most common prediction
        if action_votes:
            return max(action_votes.items(), key=lambda x: x[1])[0]
        else:
            return 'call'  # Default fallback

    def get_opponent_analysis(self, opponent_id: str) -> Dict[str, Any]:
        """Get comprehensive opponent analysis"""
        if opponent_id not in self.opponent_profiles:
            return {'error': 'Opponent not found'}

        profile = self.opponent_profiles[opponent_id]

        # Compile comprehensive analysis
        analysis = {
            'opponent_id': opponent_id,
            'statistical_profile': {
                'total_hands': profile.total_hands,
                'vpip': profile.vpip,
                'pfr': profile.pfr,
                'af': profile.af,
                'style_category': profile.style_category
            },
            'modeling_insights': {
                'sequential_patterns': self._analyze_sequential_patterns(opponent_id),
                'belief_state_summary': self._summarize_belief_state(opponent_id),
                'adaptability_assessment': self._assess_adaptability(opponent_id)
            },
            'strategic_recommendations': self._generate_strategic_recommendations(profile),
            'exploitation_opportunities': self._identify_exploitation_opportunities(profile)
        }

        return analysis

    def _analyze_sequential_patterns(self, opponent_id: str) -> Dict[str, Any]:
        """Analyze sequential patterns in opponent behavior"""
        if opponent_id not in self.action_histories:
            return {}

        actions = self.action_histories[opponent_id]

        # Analyze patterns by game phase
        phase_patterns = {}
        for action in actions[-50:]:  # Last 50 actions
            if action.game_phase not in phase_patterns:
                phase_patterns[action.game_phase] = []
            phase_patterns[action.game_phase].append(action.action_type)

        # Calculate action frequencies by phase
        pattern_analysis = {}
        for phase, phase_actions in phase_patterns.items():
            total_actions = len(phase_actions)
            action_freq = {}
            for action in phase_actions:
                action_freq[action] = action_freq.get(action, 0) + 1

            pattern_analysis[phase] = {
                freq: count / total_actions for freq, count in action_freq.items()
            }

        return pattern_analysis

    def _summarize_belief_state(self, opponent_id: str) -> Dict[str, Any]:
        """Summarize current belief state"""
        if opponent_id not in self.transformer_beliefs:
            return {}

        belief_state = self.transformer_beliefs[opponent_id]['belief_state']

        # Find strongest and weakest hands in belief
        sorted_hands = sorted(belief_state.items(), key=lambda x: x[1], reverse=True)

        return {
            'strongest_hands': sorted_hands[:5],
            'weakest_hands': sorted_hands[-5:],
            'belief_entropy': self._calculate_belief_entropy(belief_state),
            'confidence_level': max(belief_state.values())
        }

    def _calculate_belief_entropy(self, belief_state: Dict[str, float]) -> float:
        """Calculate entropy of belief distribution (uncertainty measure)"""
        entropy = 0.0
        for prob in belief_state.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)

        return entropy

    def _assess_adaptability(self, opponent_id: str) -> Dict[str, Any]:
        """Assess how much opponent's style changes over time"""
        if opponent_id not in self.action_histories or len(self.action_histories[opponent_id]) < 100:
            return {'adaptability_score': 0.0, 'trend': 'insufficient_data'}

        actions = self.action_histories[opponent_id]

        # Compare first half vs second half of actions
        mid_point = len(actions) // 2
        first_half = actions[:mid_point]
        second_half = actions[mid_point:]

        # Calculate style metrics for each half
        first_half_stats = self._calculate_style_metrics(first_half)
        second_half_stats = self._calculate_style_metrics(second_half)

        # Calculate difference (adaptability measure)
        adaptability_score = 0.0
        for metric in ['vpip', 'pfr', 'af']:
            diff = abs(first_half_stats.get(metric, 0) - second_half_stats.get(metric, 0))
            adaptability_score += diff

        return {
            'adaptability_score': adaptability_score,
            'trend': 'increasing' if adaptability_score > 0.5 else 'stable',
            'first_half_stats': first_half_stats,
            'second_half_stats': second_half_stats
        }

    def _calculate_style_metrics(self, actions: List[OpponentAction]) -> Dict[str, float]:
        """Calculate style metrics for a set of actions"""
        if not actions:
            return {}

        # Count actions by type and phase
        action_counts = {'fold': 0, 'call': 0, 'raise': 0, 'check': 0}
        phase_counts = {'preflop': 0, 'flop': 0, 'turn': 0, 'river': 0}

        for action in actions:
            action_counts[action.action_type] += 1
            phase_counts[action.game_phase] += 1

        total_actions = len(actions)

        # Calculate metrics
        metrics = {}
        if phase_counts['preflop'] > 0:
            preflop_actions = [a for a in actions if a.game_phase == 'preflop']
            preflop_voluntary = sum(1 for a in preflop_actions if a.action_type in ['call', 'raise'])
            metrics['vpip'] = preflop_voluntary / phase_counts['preflop']

            preflop_raises = sum(1 for a in preflop_actions if a.action_type == 'raise')
            metrics['pfr'] = preflop_raises / phase_counts['preflop']

        if action_counts['call'] > 0:
            metrics['af'] = action_counts['raise'] / action_counts['call']

        return metrics

    def _generate_strategic_recommendations(self, profile: OpponentProfile) -> List[str]:
        """Generate strategic recommendations based on opponent profile"""
        recommendations = []

        if profile.style_category == 'TAG':
            recommendations.append("Play solid value hands, avoid marginal spots")
            recommendations.append("Consider 3-betting light for balance")
            recommendations.append("Respect continuation bets on most boards")

        elif profile.style_category == 'LAG':
            recommendations.append("Trap with strong hands, let them bluff")
            recommendations.append("Widen calling ranges against their aggression")
            recommendations.append("Look for spots to punish their wide ranges")

        elif profile.style_category == 'NIT':
            recommendations.append("Steal blinds aggressively")
            recommendations.append("Value bet thinly for maximum profit")
            recommendations.append("Bluff more frequently against their tight range")

        elif profile.style_category == 'FISH':
            recommendations.append("Value bet relentlessly with any decent hand")
            recommendations.append("Avoid fancy plays, keep it simple")
            recommendations.append("Extract maximum value from made hands")

        return recommendations

    def _identify_exploitation_opportunities(self, profile: OpponentProfile) -> List[str]:
        """Identify specific opportunities to exploit opponent weaknesses"""
        opportunities = []

        if profile.af < 0.8:  # Too passive
            opportunities.append("Bluff more frequently, especially on later streets")
            opportunities.append("Bet smaller for value to induce calls")

        if profile.pfr < 0.05:  # Too tight preflop
            opportunities.append("Steal blinds and raise limpers aggressively")
            opportunities.append("3-bet wider range for value")

        if profile.vpip > 0.30:  # Too loose
            opportunities.append("Tighten up and wait for premium hands")
            opportunities.append("Punish their wide ranges with strong value bets")

        return opportunities

    def get_modeling_performance(self, opponent_id: str) -> Dict[str, Any]:
        """Get performance metrics for opponent modeling"""
        if opponent_id not in self.modeling_accuracy:
            return {'accuracy': 0.0, 'sample_size': 0}

        accuracy_history = self.modeling_accuracy[opponent_id]

        if not accuracy_history:
            return {'accuracy': 0.0, 'sample_size': 0}

        return {
            'accuracy': sum(accuracy_history) / len(accuracy_history),
            'sample_size': len(accuracy_history),
            'recent_accuracy': sum(accuracy_history[-10:]) / min(10, len(accuracy_history)),
            'trend': self._calculate_accuracy_trend(accuracy_history)
        }

    def _calculate_accuracy_trend(self, accuracy_history: List[float]) -> str:
        """Calculate trend in modeling accuracy"""
        if len(accuracy_history) < 10:
            return 'insufficient_data'

        first_half = accuracy_history[:len(accuracy_history)//2]
        second_half = accuracy_history[len(accuracy_history)//2:]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        if second_avg > first_avg + 0.05:
            return 'improving'
        elif second_avg < first_avg - 0.05:
            return 'declining'
        else:
            return 'stable'

    def reset_opponent_model(self, opponent_id: str) -> None:
        """Reset opponent model (for new opponent or session)"""
        if opponent_id in self.opponent_profiles:
            del self.opponent_profiles[opponent_id]
        if opponent_id in self.action_histories:
            del self.action_histories[opponent_id]
        if opponent_id in self.modeling_accuracy:
            del self.modeling_accuracy[opponent_id]
        if opponent_id in self.transformer_beliefs:
            del self.transformer_beliefs[opponent_id]

        logger.info(f"Reset opponent model for {opponent_id}")

    def get_all_opponent_profiles(self) -> Dict[str, OpponentProfile]:
        """Get all opponent profiles for analysis"""
        return self.opponent_profiles.copy()

    def export_opponent_data(self, opponent_id: str) -> Dict[str, Any]:
        """Export comprehensive opponent data for external analysis"""
        if opponent_id not in self.opponent_profiles:
            return {'error': 'Opponent not found'}

        return {
            'profile': self.opponent_profiles[opponent_id],
            'action_history': self.action_histories.get(opponent_id, []),
            'belief_state': self.transformer_beliefs.get(opponent_id, {}),
            'modeling_performance': self.get_modeling_performance(opponent_id),
            'analysis': self.get_opponent_analysis(opponent_id),
            'export_timestamp': time.time()
        }