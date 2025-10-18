"""
Poker odds calculation algorithms
"""
import logging
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import math
import itertools
from .poker_detector import Card, PokerGameState
from .opponent_modeler import AdvancedOpponentModeler, OpponentAction

logger = logging.getLogger(__name__)


@dataclass
class PokerOdds:
    """Calculated poker odds and probabilities"""
    hand_strength: float  # 0-1, probability of having best hand
    hand_rank: str  # 'high_card', 'pair', 'two_pair', etc.
    hand_category: str  # 'monster', 'strong', 'medium', 'weak', 'trash'
    equity_vs_random: float  # Equity vs random hand
    equity_vs_range: float  # Equity vs opponent's perceived range
    pot_odds: float  # Current pot odds
    recommended_action: str  # 'fold', 'call', 'raise'
    outs: int  # Number of outs to improve
    drawing_to: List[str]  # What you're drawing to

    # Enhanced fields
    specific_hand_odds: Dict[str, float]  # Odds of making specific hands
    expected_value: float  # Expected value of current decision
    opponent_range_analysis: Dict[str, float]  # Analysis of opponent's likely range
    position_advantage: float  # Positional advantage factor
    bluff_factor: float  # Likelihood opponent is bluffing
    fold_equity: float  # Equity gained when opponent folds


class PokerStoveEngine:
    """PokerStove-style equity calculation engine"""

    def __init__(self):
        self.hand_range_cache = {}
        self.equity_cache = {}

    def calculate_range_equity(self, hand_range1: List[str], hand_range2: List[str],
                             board: List[str] = None) -> float:
        """Calculate equity between two hand ranges (PokerStove style)"""
        if board is None:
            board = []

        # Create cache key
        cache_key = (frozenset(hand_range1), frozenset(hand_range2), frozenset(board))

        if cache_key in self.equity_cache:
            return self.equity_cache[cache_key]

        # Simplified PokerStove-style calculation
        # In reality, this would use more sophisticated enumeration
        equity = 0.5  # Placeholder

        # Adjust based on hand strength vs ranges
        if hand_range1 and hand_range2:
            # Simple range vs range analysis
            avg_strength1 = self._calculate_range_strength(hand_range1)
            avg_strength2 = self._calculate_range_strength(hand_range2)

            if avg_strength1 > avg_strength2:
                equity = 0.6
            elif avg_strength1 < avg_strength2:
                equity = 0.4

        self.equity_cache[cache_key] = equity
        return equity

    def _calculate_range_strength(self, hand_range: List[str]) -> float:
        """Calculate average strength of a hand range"""
        if not hand_range:
            return 0.5

        # Simplified range strength calculation
        strength_values = {
            'AA': 1.0, 'KK': 0.95, 'QQ': 0.9, 'JJ': 0.85,
            'AK': 0.8, 'AQ': 0.75, 'AJ': 0.7, 'KQ': 0.65,
            '22': 0.3, '32': 0.1, '72': 0.05
        }

        total_strength = 0.0
        for hand in hand_range[:10]:  # Sample first 10 hands
            total_strength += strength_values.get(hand, 0.5)

        return total_strength / min(len(hand_range), 10)


class PokerOddsCalculator:
    """Calculates poker odds and probabilities with POMDP-aware uncertainty modeling"""

    def __init__(self):
        # Standard deck composition
        self.ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
        self.suits = ['hearts', 'diamonds', 'clubs', 'spades']
        self.deck = self._create_deck()

        # Advanced opponent modeling (expert analysis)
        self.opponent_modeler = AdvancedOpponentModeler()

        # POMDP-aware uncertainty modeling (from expert analysis)
        self.perception_uncertainty = 0.02  # 2% base uncertainty in card detection
        self.belief_state_tracking = {}     # Track belief states over time
        self.observation_history = []       # Store action observations for belief updates

        # Enhanced opponent range analysis with modeling
        self.dynamic_opponent_ranges = {}   # Opponent-specific ranges based on modeling

        # Hand rankings (higher is better)
        self.hand_rankings = {
            'high_card': 1,
            'pair': 2,
            'two_pair': 3,
            'three_of_a_kind': 4,
            'straight': 5,
            'flush': 6,
            'full_house': 7,
            'four_of_a_kind': 8,
            'straight_flush': 9,
            'royal_flush': 10
        }

    def _create_deck(self) -> List[Card]:
        """Create a standard 52-card deck"""
        deck = []
        for rank in self.ranks:
            for suit in self.suits:
                deck.append(Card(rank=rank, suit=suit, confidence=1.0))
        return deck

    def calculate_odds(self, game_state: PokerGameState) -> Optional[PokerOdds]:
        """Calculate comprehensive poker odds for current game state"""
        try:
            # Convert detected cards to standard format
            player_cards = game_state.player_cards
            community_cards = game_state.community_cards

            if not player_cards:
                return None

            # Evaluate current hand strength
            hand_rank, hand_category = self._evaluate_hand(player_cards, community_cards)

            # Calculate equity against random hands
            equity_random = self._calculate_equity_vs_random(player_cards, community_cards)

            # Calculate equity against opponent ranges (simplified)
            equity_range = self._calculate_equity_vs_range(player_cards, community_cards)

            # Calculate comprehensive odds
            pot_odds = self._calculate_pot_odds(game_state)

            # Determine outs and drawing hands
            outs, drawing_to = self._calculate_outs(player_cards, community_cards)

            # Calculate specific hand odds
            specific_hand_odds = self._calculate_specific_hand_odds(player_cards, community_cards)

            # Calculate expected value and advanced metrics
            expected_value = self._calculate_expected_value(equity_random, pot_odds, game_state)
            opponent_range_analysis = self._analyze_opponent_range(game_state)
            position_advantage = self._calculate_position_advantage(game_state)
            bluff_factor = self._calculate_bluff_factor(game_state)
            fold_equity = self._calculate_fold_equity(game_state)

            # Recommend action based on comprehensive calculations
            recommended_action = self._recommend_advanced_action(
                equity_random, pot_odds, hand_category, game_state,
                expected_value, position_advantage, bluff_factor
            )

            return PokerOdds(
                hand_strength=equity_random,
                hand_rank=hand_rank,
                hand_category=hand_category,
                equity_vs_random=equity_random,
                equity_vs_range=equity_range,
                pot_odds=pot_odds,
                recommended_action=recommended_action,
                outs=outs,
                drawing_to=drawing_to,
                specific_hand_odds=specific_hand_odds,
                expected_value=expected_value,
                opponent_range_analysis=opponent_range_analysis,
                position_advantage=position_advantage,
                bluff_factor=bluff_factor,
                fold_equity=fold_equity
            )

        except Exception as e:
            logger.error(f"Error calculating poker odds: {e}", exc_info=True)
            return None

    def _evaluate_hand(self, player_cards: List[Card], community_cards: List[Card]) -> Tuple[str, str]:
        """Evaluate the current hand strength"""
        all_cards = player_cards + community_cards
        if len(all_cards) < 5:
            # Not enough cards for full evaluation
            return 'incomplete', 'unknown'

        # Convert to simple format for evaluation
        hand = self._convert_cards_for_evaluation(all_cards)

        # Check for various hand types (simplified implementation)
        if self._is_royal_flush(hand):
            return 'royal_flush', 'monster'
        elif self._is_straight_flush(hand):
            return 'straight_flush', 'monster'
        elif self._is_four_of_a_kind(hand):
            return 'four_of_a_kind', 'monster'
        elif self._is_full_house(hand):
            return 'full_house', 'strong'
        elif self._is_flush(hand):
            return 'flush', 'strong'
        elif self._is_straight(hand):
            return 'straight', 'medium'
        elif self._is_three_of_a_kind(hand):
            return 'three_of_a_kind', 'medium'
        elif self._is_two_pair(hand):
            return 'two_pair', 'weak'
        elif self._is_pair(hand):
            return 'pair', 'weak'
        else:
            return 'high_card', 'trash'

    def _convert_cards_for_evaluation(self, cards: List[Card]) -> List[Tuple[str, str]]:
        """Convert Card objects to simple tuples for evaluation"""
        return [(card.rank, card.suit) for card in cards]

    def _is_royal_flush(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is a royal flush"""
        # Simplified check
        ranks = sorted([self._rank_to_value(r[0]) for r in hand])
        return (self._is_straight(hand) and self._is_flush(hand) and
                ranks == [10, 11, 12, 13, 14])  # 10,J,Q,K,A

    def _is_straight_flush(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is a straight flush"""
        return self._is_straight(hand) and self._is_flush(hand)

    def _is_four_of_a_kind(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is four of a kind"""
        ranks = [r[0] for r in hand]
        for rank in set(ranks):
            if ranks.count(rank) >= 4:
                return True
        return False

    def _is_full_house(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is a full house"""
        ranks = [r[0] for r in hand]
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        counts = sorted(rank_counts.values())
        return counts == [2, 3]

    def _is_flush(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is a flush"""
        suits = [s[1] for s in hand]
        return len(set(suits)) == 1

    def _is_straight(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is a straight"""
        ranks = sorted(set([self._rank_to_value(r[0]) for r in hand]))
        if len(ranks) < 5:
            return False

        # Check for regular straight
        for i in range(len(ranks) - 4):
            if ranks[i+4] - ranks[i] == 4:
                return True

        # Check for A-2-3-4-5 straight (wheel)
        if set(ranks) == {2, 3, 4, 5, 14}:  # A,2,3,4,5
            return True

        return False

    def _is_three_of_a_kind(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is three of a kind"""
        ranks = [r[0] for r in hand]
        for rank in set(ranks):
            if ranks.count(rank) >= 3:
                return True
        return False

    def _is_two_pair(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is two pair"""
        ranks = [r[0] for r in hand]
        rank_counts = {}
        for rank in ranks:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        pairs = [count for count in rank_counts.values() if count >= 2]
        return len(pairs) >= 2

    def _is_pair(self, hand: List[Tuple[str, str]]) -> bool:
        """Check if hand is a pair"""
        ranks = [r[0] for r in hand]
        for rank in set(ranks):
            if ranks.count(rank) >= 2:
                return True
        return False

    def _rank_to_value(self, rank: str) -> int:
        """Convert rank string to numeric value"""
        rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        return rank_values.get(rank, 0)

    def _calculate_equity_vs_random(self, player_cards: List[Card], community_cards: List[Card]) -> float:
        """Calculate equity against random opponent hands with enhanced Monte Carlo"""
        # Adaptive simulation count based on configuration
        base_simulations = getattr(self, 'simulation_count', 10000)
        total_simulations = min(base_simulations, 50000)  # Cap for performance

        wins = 0
        ties = 0
        known_cards = player_cards + community_cards
        remaining_deck = self._get_remaining_deck(known_cards)

        # Pre-compute deck length for efficiency
        deck_size = len(remaining_deck)

        for _ in range(total_simulations):
            # Deal random opponent cards more efficiently
            if deck_size < 2:
                break

            opponent_indices = np.random.choice(deck_size, size=2, replace=False)
            opponent_cards = [remaining_deck[i] for i in opponent_indices]

            # Remove opponent cards from remaining deck
            remaining_deck_temp = [card for i, card in enumerate(remaining_deck) if i not in opponent_indices]

            # Deal remaining community cards if needed
            needed_community = 5 - len(community_cards)
            if needed_community > 0:
                if len(remaining_deck_temp) < needed_community:
                    continue
                community_indices = np.random.choice(len(remaining_deck_temp), size=needed_community, replace=False)
                current_community = community_cards + [remaining_deck_temp[i] for i in community_indices]
            else:
                current_community = community_cards

            # Evaluate both hands with enhanced comparison
            player_hand_value = self._evaluate_hand_value(player_cards, current_community)
            opponent_hand_value = self._evaluate_hand_value(opponent_cards, current_community)

            if player_hand_value > opponent_hand_value:
                wins += 1
            elif player_hand_value == opponent_hand_value:
                ties += 1

        total_valid = wins + ties
        equity = (wins + 0.5 * ties) / total_simulations if total_simulations > 0 else 0.0

        # Cache result for performance
        cache_key = f"random_{len(player_cards)}_{len(community_cards)}"
        self._cache_equity_result(cache_key, equity)

        return equity

    def _calculate_equity_vs_range(self, player_cards: List[Card], community_cards: List[Card],
                                 opponent_range: List[str] = None) -> float:
        """Calculate equity against specific opponent range"""
        if not opponent_range:
            # Use default tight range if no range provided
            opponent_range = ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77',
                            'AKs', 'AQs', 'AJs', 'AKo', 'AQo']

        total_simulations = getattr(self, 'simulation_count', 10000) // 10  # Fewer simulations for range analysis
        wins = 0
        ties = 0

        known_cards = player_cards + community_cards
        remaining_deck = self._get_remaining_deck(known_cards)

        for _ in range(total_simulations):
            # Select random hand from opponent's range
            opponent_hand_str = np.random.choice(opponent_range)
            opponent_cards = self._string_hand_to_cards(opponent_hand_str, remaining_deck)

            if not opponent_cards:
                continue

            # Remove opponent cards from deck
            remaining_deck_temp = [card for card in remaining_deck if card not in opponent_cards]

            # Deal remaining community cards
            needed_community = 5 - len(community_cards)
            if needed_community > 0:
                if len(remaining_deck_temp) < needed_community:
                    continue
                community_indices = np.random.choice(len(remaining_deck_temp), size=needed_community, replace=False)
                current_community = community_cards + [remaining_deck_temp[i] for i in community_indices]
            else:
                current_community = community_cards

            # Evaluate hands
            player_hand_value = self._evaluate_hand_value(player_cards, current_community)
            opponent_hand_value = self._evaluate_hand_value(opponent_cards, current_community)

            if player_hand_value > opponent_hand_value:
                wins += 1
            elif player_hand_value == opponent_hand_value:
                ties += 1

        return (wins + 0.5 * ties) / total_simulations if total_simulations > 0 else 0.0

    def _string_hand_to_cards(self, hand_str: str, available_cards: List[Card]) -> List[Card]:
        """Convert hand string (e.g., 'AKs') to Card objects"""
        if len(hand_str) < 2:
            return []

        rank1, rank2 = hand_str[0], hand_str[1]
        suited = len(hand_str) > 2 and hand_str[2] == 's'

        # Find matching cards in available deck
        card1 = None
        card2 = None

        for card in available_cards:
            if card.rank == rank1:
                if suited:
                    # For suited hands, prefer same suit
                    if card1 is None:
                        card1 = card
                    elif card.suit == card1.suit and card2 is None:
                        card2 = card
                        break
                else:
                    # For offsuit, take first available
                    if card1 is None:
                        card1 = card
                    elif card2 is None:
                        card2 = card
                        break

        return [card1, card2] if card1 and card2 else []

    def calculate_effective_hand_strength(self, player_cards: List[Card],
                                         community_cards: List[Card],
                                         opponent_range: List[str] = None) -> float:
        """Calculate Effective Hand Strength (EHS) with POMDP-aware uncertainty"""
        if not opponent_range:
            opponent_range = ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo']

        # POMDP-aware calculation: Account for perception uncertainty
        current_equity = self._calculate_equity_vs_range(player_cards, community_cards, opponent_range)

        # Adjust equity for perception uncertainty (expert analysis insight)
        uncertainty_adjusted_equity = current_equity * (1 - self.perception_uncertainty)

        # Calculate PPOT (Positive Potential) and NPOT (Negative Potential)
        ppot = self._calculate_ppot(player_cards, community_cards, opponent_range)
        npot = self._calculate_npot(player_cards, community_cards, opponent_range)

        # Enhanced EHS with uncertainty consideration
        ehs = uncertainty_adjusted_equity * (1 - npot) + (1 - uncertainty_adjusted_equity) * ppot

        # Update belief state tracking (POMDP approach)
        self._update_belief_state(player_cards, community_cards, opponent_range, ehs)

        return ehs

    def _update_belief_state(self, player_cards: List[Card], community_cards: List[Card],
                           opponent_range: List[str], current_ehs: float) -> None:
        """Update belief state using POMDP framework (expert analysis)"""

        # Create belief state key
        belief_key = self._generate_belief_key(player_cards, community_cards)

        # Initialize belief state if new
        if belief_key not in self.belief_state_tracking:
            self.belief_state_tracking[belief_key] = {
                'initial_belief': self._initialize_uniform_belief(opponent_range),
                'observations': [],
                'updated_belief': None,
                'last_update': time.time()
            }

        # Store current observation for future belief updates
        observation = {
            'timestamp': time.time(),
            'community_cards': len(community_cards),
            'current_ehs': current_ehs,
            'perception_confidence': 1 - self.perception_uncertainty
        }
        self.belief_state_tracking[belief_key]['observations'].append(observation)

        # Keep only recent observations (last 20)
        if len(self.belief_state_tracking[belief_key]['observations']) > 20:
            self.belief_state_tracking[belief_key]['observations'] = \
                self.belief_state_tracking[belief_key]['observations'][-20:]

    def _initialize_uniform_belief(self, opponent_range: List[str]) -> Dict[str, float]:
        """Initialize uniform belief distribution over opponent range"""
        if not opponent_range:
            return {}

        uniform_prob = 1.0 / len(opponent_range)
        return {hand: uniform_prob for hand in opponent_range}

    def _generate_belief_key(self, player_cards: List[Card], community_cards: List[Card]) -> str:
        """Generate unique key for belief state tracking"""
        player_ranks = sorted([card.rank for card in player_cards])
        community_count = len(community_cards)
        return f"belief_{player_ranks}_{community_count}_{time.time() // 60}"  # Per-minute tracking

    def _calculate_ppot(self, player_cards: List[Card], community_cards: List[Card],
                       opponent_range: List[str]) -> float:
        """Calculate Positive Potential (probability hand improves to become best)"""
        # Simplified PPOT calculation
        needed_cards = 5 - len(community_cards)
        if needed_cards <= 0:
            return 0.0

        # This is a simplified version - full implementation would require
        # simulating all possible future cards and opponent responses
        base_ppot = 0.3 if needed_cards > 0 else 0.0

        # Adjust based on current hand strength and cards needed
        current_strength = self._calculate_equity_vs_range(player_cards, community_cards, opponent_range)
        if current_strength < 0.4:  # Behind - more potential to improve
            base_ppot *= 1.5
        elif current_strength > 0.7:  # Ahead - less potential to improve
            base_ppot *= 0.5

        return min(0.8, base_ppot)

    def _calculate_npot(self, player_cards: List[Card], community_cards: List[Card],
                       opponent_range: List[str]) -> float:
        """Calculate Negative Potential (probability hand gets outdrawn)"""
        # Simplified NPOT calculation
        needed_cards = 5 - len(community_cards)
        if needed_cards <= 0:
            return 0.0

        current_strength = self._calculate_equity_vs_range(player_cards, community_cards, opponent_range)

        # If we're ahead, there's potential to be outdrawn
        base_npot = current_strength * 0.6 if current_strength > 0.5 else 0.2

        return min(0.7, base_npot)

    def _cache_equity_result(self, cache_key: str, result: float) -> None:
        """Cache equity calculation results for performance"""
        # Simple in-memory cache (in production, use Redis or similar)
        if not hasattr(self, '_equity_cache'):
            self._equity_cache = {}

        self._equity_cache[cache_key] = {
            'result': result,
            'timestamp': time.time()
        }

        # Clean old cache entries (older than 5 minutes)
        current_time = time.time()
        self._equity_cache = {
            k: v for k, v in self._equity_cache.items()
            if current_time - v['timestamp'] < 300
        }

    def _calculate_equity_vs_range(self, player_cards: List[Card], community_cards: List[Card]) -> float:
        """Calculate equity against opponent's perceived range"""
        # Simplified - assume opponent plays reasonably tight
        # In reality, this would use more sophisticated range analysis
        return self._calculate_equity_vs_random(player_cards, community_cards) * 0.8

    def _calculate_pot_odds(self, game_state: PokerGameState) -> float:
        """Calculate current pot odds"""
        if not game_state.pot_size or not game_state.current_bet:
            return 0.0

        # Pot odds = amount to call / (pot size + amount to call)
        return game_state.current_bet / (game_state.pot_size + game_state.current_bet)

    def _calculate_specific_hand_odds(self, player_cards: List[Card], community_cards: List[Card]) -> Dict[str, float]:
        """Calculate odds of making specific hands"""
        specific_odds = {
            'royal_flush': 0.0,
            'straight_flush': 0.0,
            'four_of_a_kind': 0.0,
            'full_house': 0.0,
            'flush': 0.0,
            'straight': 0.0,
            'three_of_a_kind': 0.0,
            'two_pair': 0.0,
            'pair': 0.0
        }

        try:
            # Get remaining deck
            known_cards = player_cards + community_cards
            remaining_deck = self._get_remaining_deck(known_cards)

            if not remaining_deck:
                return specific_odds

            # Calculate odds for each hand type
            total_simulations = 1000
            made_hands = {hand_type: 0 for hand_type in specific_odds.keys()}

            for _ in range(total_simulations):
                # Deal remaining community cards if needed
                needed_cards = 5 - len(community_cards)
                if needed_cards > 0:
                    sim_community = community_cards + self._deal_random_cards(remaining_deck, needed_cards)
                else:
                    sim_community = community_cards

                # Evaluate hand
                hand_value = self._evaluate_hand_value(player_cards, sim_community)
                hand_type = self._get_hand_type_from_value(hand_value)

                if hand_type in made_hands:
                    made_hands[hand_type] += 1

            # Convert to probabilities
            for hand_type in specific_odds:
                specific_odds[hand_type] = made_hands[hand_type] / total_simulations

        except Exception as e:
            logger.debug(f"Error calculating specific hand odds: {e}")

        return specific_odds

    def _get_hand_type_from_value(self, hand_value: int) -> str:
        """Get hand type string from hand value"""
        if hand_value >= 10000:
            return 'royal_flush'
        elif hand_value >= 9000:
            return 'straight_flush'
        elif hand_value >= 8000:
            return 'four_of_a_kind'
        elif hand_value >= 7000:
            return 'full_house'
        elif hand_value >= 6000:
            return 'flush'
        elif hand_value >= 5000:
            return 'straight'
        elif hand_value >= 4000:
            return 'three_of_a_kind'
        elif hand_value >= 3000:
            return 'two_pair'
        elif hand_value >= 2000:
            return 'pair'
        else:
            return 'high_card'

    def _calculate_expected_value(self, equity: float, pot_odds: float, game_state: PokerGameState) -> float:
        """Calculate expected value of current decision"""
        if not game_state.current_bet or not game_state.pot_size:
            return 0.0

        # EV = (equity * pot_size) - ((1 - equity) * bet_amount)
        bet_amount = game_state.current_bet
        ev_call = (equity * game_state.pot_size) - ((1 - equity) * bet_amount)

        return ev_call

    def _analyze_opponent_range(self, game_state: PokerGameState) -> Dict[str, float]:
        """Enhanced opponent range analysis using advanced modeling (expert analysis)"""

        # Get opponent information from game state
        opponent_positions = [pos for pos in ['BTN', 'CO', 'HJ', 'LJ', 'MP', 'UTG', 'SB', 'BB']
                            if pos != game_state.player_position]

        range_analysis = {}

        for opponent_pos in opponent_positions:
            opponent_id = f"opponent_{opponent_pos}"

            # Use advanced opponent modeling if available
            if opponent_id in self.opponent_modeler.opponent_profiles:
                profile = self.opponent_modeler.opponent_profiles[opponent_id]

                # Generate dynamic range based on opponent profile
                dynamic_range = self._generate_dynamic_opponent_range(profile, game_state, opponent_pos)

                range_analysis[opponent_pos] = {
                    'style_category': profile.style_category,
                    'vpip': profile.vpip,
                    'pfr': profile.pfr,
                    'af': profile.af,
                    'estimated_range': dynamic_range,
                    'exploitation_opportunities': self.opponent_modeler._identify_exploitation_opportunities(profile)
                }
            else:
                # Fallback to basic analysis
                range_analysis[opponent_pos] = {
                    'style_category': 'unknown',
                    'estimated_range': self._get_default_opponent_range(opponent_pos),
                    'exploitation_opportunities': []
                }

        return range_analysis

    def _generate_dynamic_opponent_range(self, profile, game_state: PokerGameState, position: str) -> List[str]:
        """Generate dynamic opponent range based on advanced modeling"""

        # Base ranges by style category (expert analysis)
        style_ranges = {
            'TAG': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', 'AKs', 'AQs', 'AKo', 'AQo'],
            'LAG': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22',
                   'AKs', 'AQs', 'AJs', 'ATs', 'AKo', 'AQo', 'AJo', 'ATo', 'KQs', 'KJs'],
            'NIT': ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo'],
            'FISH': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', '55', '44', '33', '22',
                    'AKs', 'AQs', 'AKo', 'AQo', 'KQs'],
            'MANIAC': ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', '88', '77', '66', 'AKs', 'AKo']
        }

        base_range = style_ranges.get(profile.style_category, style_ranges['TAG'])

        # Adjust range based on position and game phase
        adjusted_range = self._adjust_range_for_context(base_range, position, game_state.game_phase)

        return adjusted_range

    def _adjust_range_for_context(self, base_range: List[str], position: str, game_phase: str) -> List[str]:
        """Adjust opponent range based on positional and situational context"""

        adjusted_range = base_range.copy()

        # Positional adjustments
        position_multipliers = {
            'BTN': 1.3, 'CO': 1.2, 'HJ': 1.1, 'LJ': 1.0, 'MP': 0.9, 'UTG': 0.8, 'SB': 0.7, 'BB': 0.6
        }

        multiplier = position_multipliers.get(position, 1.0)

        # Phase adjustments
        if game_phase == 'preflop':
            # Widen range in later positions
            if multiplier > 1.0:
                # Add more speculative hands for loose positions
                speculative_hands = ['A9s', 'A8s', 'A7s', 'KJs', 'KTs', 'QJs']
                adjusted_range.extend(speculative_hands[:int(len(speculative_hands) * (multiplier - 1.0))])

        return list(set(adjusted_range))  # Remove duplicates

    def _get_default_opponent_range(self, position: str) -> List[str]:
        """Get default opponent range when no modeling data available"""
        # Conservative default ranges by position
        default_ranges = {
            'BTN': ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo'],
            'CO': ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo'],
            'HJ': ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AKo'],
            'LJ': ['AA', 'KK', 'QQ', 'AKs', 'AKo'],
            'MP': ['AA', 'KK', 'AKs', 'AKo'],
            'UTG': ['AA', 'KK', 'AKs', 'AKo'],
            'SB': ['AA', 'KK', 'AKs'],
            'BB': ['AA', 'KK', 'AKs']
        }

        return default_ranges.get(position, ['AA', 'KK', 'AKs'])

    def track_opponent_action(self, opponent_id: str, action: str, bet_size: float = 0.0,
                            position: str = 'unknown', game_phase: str = 'unknown',
                            pot_size: float = 0.0, stack_size: float = 0.0) -> None:
        """Track opponent action for advanced modeling (expert analysis)"""

        opponent_action = OpponentAction(
            timestamp=time.time(),
            action_type=action,
            bet_size=bet_size,
            position=position,
            game_phase=game_phase,
            pot_size=pot_size,
            stack_size=stack_size
        )

        # Update advanced opponent modeler
        self.opponent_modeler.update_opponent_model(opponent_id, opponent_action)

    def _calculate_position_advantage(self, game_state: PokerGameState) -> float:
        """Calculate positional advantage factor"""
        position_values = {
            'BTN': 1.0,    # Button - best position
            'CO': 0.9,     # Cutoff
            'HJ': 0.7,     # Hijack
            'LJ': 0.5,     # Lojack
            'MP': 0.4,     # Middle position
            'UTG+1': 0.3,  # UTG + 1
            'UTG': 0.2,    # Under the gun - worst position
            'SB': 0.1,     # Small blind
            'BB': 0.0      # Big blind - worst
        }

        return position_values.get(game_state.player_position or 'UTG', 0.5)

    def _calculate_bluff_factor(self, game_state: PokerGameState) -> float:
        """Calculate likelihood opponent is bluffing"""
        bluff_factor = 0.1  # Base bluff factor

        # Increase bluff factor in later positions
        if game_state.player_position in ['BTN', 'CO']:
            bluff_factor += 0.1

        # Increase bluff factor when pot is large relative to stacks
        if game_state.pot_size and game_state.player_stack:
            pot_stack_ratio = game_state.pot_size / game_state.player_stack
            if pot_stack_ratio > 0.5:
                bluff_factor += 0.1

        return min(bluff_factor, 0.5)  # Cap at 50%

    def _calculate_fold_equity(self, game_state: PokerGameState) -> float:
        """Calculate equity gained when opponent folds"""
        if not game_state.current_bet:
            return 0.0

        # Fold equity = probability opponent folds * pot size / bet size
        fold_probability = self._estimate_fold_probability(game_state)
        pot_odds_ratio = game_state.pot_size / game_state.current_bet if game_state.current_bet > 0 else 0

        return fold_probability * pot_odds_ratio

    def _estimate_fold_probability(self, game_state: PokerGameState) -> float:
        """Estimate probability opponent will fold to a bet"""
        base_fold_prob = 0.3  # Base fold probability

        # Adjust based on opponent tendencies
        if game_state.player_tendencies:
            for tendency in game_state.player_tendencies.values():
                if tendency == 'tight':
                    base_fold_prob += 0.2
                elif tendency == 'loose':
                    base_fold_prob -= 0.1

        # Adjust based on bet sizing
        if game_state.current_bet and game_state.player_stack:
            bet_size_ratio = game_state.current_bet / game_state.player_stack
            if bet_size_ratio > 0.3:  # Large bet
                base_fold_prob += 0.2

        return max(0.1, min(0.8, base_fold_prob))  # Keep within reasonable bounds

    def _recommend_advanced_action(self, equity: float, pot_odds: float, hand_category: str,
                                 game_state: PokerGameState, expected_value: float,
                                 position_advantage: float, bluff_factor: float) -> str:
        """Advanced action recommendation considering all factors"""
        # Base recommendation from hand strength
        if hand_category in ['monster', 'strong']:
            base_action = 'raise'
        elif hand_category == 'medium':
            base_action = 'call' if equity > pot_odds else 'fold'
        elif hand_category == 'weak':
            base_action = 'call' if equity > pot_odds + 0.1 else 'fold'
        else:
            base_action = 'fold'

        # Adjust for position
        if position_advantage > 0.7 and base_action == 'call':
            base_action = 'raise'  # Use position to be more aggressive

        # Adjust for bluff factor
        if bluff_factor > 0.3 and base_action == 'call' and expected_value > 0:
            # Consider semi-bluff opportunities
            if hand_category in ['weak', 'medium']:
                base_action = 'raise'

        # Consider fold equity for bluffs
        if expected_value < 0 and bluff_factor > 0.2 and position_advantage > 0.5:
            # Bluff opportunity
            base_action = 'raise'

        # Final adjustments based on stack sizes and tournament situation
        if game_state.player_stack and game_state.current_bet:
            stack_remaining = game_state.player_stack - game_state.current_bet
            if stack_remaining < game_state.player_stack * 0.1:  # Short stack
                if equity > 0.3:  # Any decent hand
                    base_action = 'all_in'

        return base_action


class PokerModelTracker:
    """Tracks poker model performance and improves recommendations over time"""

    def __init__(self):
        self.history_file = 'poker_model_history.json'
        self.hand_history = []
        self.model_performance = {
            'total_hands': 0,
            'correct_predictions': 0,
            'accuracy_by_category': {},
            'profit_loss_tracking': []
        }
        self._load_history()

    def _load_history(self) -> None:
        """Load historical data from file"""
        try:
            import json
            with open(self.history_file, 'r') as f:
                data = json.load(f)
                self.hand_history = data.get('hand_history', [])
                self.model_performance = data.get('model_performance', self.model_performance)
        except (FileNotFoundError, json.JSONDecodeError):
            # Start with empty history
            pass

    def _save_history(self) -> None:
        """Save historical data to file"""
        try:
            import json
            data = {
                'hand_history': self.hand_history,
                'model_performance': self.model_performance
            }
            with open(self.history_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    def record_hand_result(self, game_state: PokerGameState, predicted_action: str,
                          actual_result: str, profit_loss: float) -> None:
        """Record the result of a hand for model improvement"""
        hand_record = {
            'timestamp': time.time(),
            'hand_strength': game_state.player_cards,  # Simplified
            'predicted_action': predicted_action,
            'actual_result': actual_result,
            'profit_loss': profit_loss,
            'pot_size': game_state.pot_size,
            'position': game_state.player_position,
            'game_phase': game_state.game_phase
        }

        self.hand_history.append(hand_record)
        self.model_performance['total_hands'] += 1

        # Update accuracy tracking
        if predicted_action == actual_result:
            self.model_performance['correct_predictions'] += 1

        # Track profit/loss
        self.model_performance['profit_loss_tracking'].append(profit_loss)

        # Update category accuracy
        category = self._categorize_hand(game_state.player_cards)
        if category not in self.model_performance['accuracy_by_category']:
            self.model_performance['accuracy_by_category'][category] = {'correct': 0, 'total': 0}

        self.model_performance['accuracy_by_category'][category]['total'] += 1
        if predicted_action == actual_result:
            self.model_performance['accuracy_by_category'][category]['correct'] += 1

        # Keep only recent history (last 1000 hands)
        if len(self.hand_history) > 1000:
            self.hand_history = self.hand_history[-1000:]

        self._save_history()

    def _categorize_hand(self, player_cards: List[Card]) -> str:
        """Categorize hand for tracking purposes"""
        # Simplified categorization - in reality would use full hand evaluation
        if len(player_cards) >= 2:
            ranks = sorted([self._rank_to_value(card.rank) for card in player_cards], reverse=True)
            if ranks[0] >= 12:  # A, K, Q
                return 'premium'
            elif ranks[0] >= 10:  # J, 10
                return 'good'
            elif ranks[0] - ranks[1] <= 2:  # Connected/suited potential
                return 'speculative'
            else:
                return 'marginal'

        return 'unknown'

    def get_model_accuracy(self) -> float:
        """Get current model accuracy"""
        if self.model_performance['total_hands'] == 0:
            return 0.0

        return self.model_performance['correct_predictions'] / self.model_performance['total_hands']

    def get_profitability_metrics(self) -> Dict[str, float]:
        """Get profitability metrics"""
        if not self.model_performance['profit_loss_tracking']:
            return {'total_profit': 0.0, 'avg_profit_per_hand': 0.0, 'win_rate': 0.0}

        total_profit = sum(self.model_performance['profit_loss_tracking'])
        hands_played = len(self.model_performance['profit_loss_tracking'])
        winning_hands = sum(1 for pl in self.model_performance['profit_loss_tracking'] if pl > 0)

        return {
            'total_profit': total_profit,
            'avg_profit_per_hand': total_profit / hands_played,
            'win_rate': winning_hands / hands_played
        }

    def get_category_accuracy(self) -> Dict[str, float]:
        """Get accuracy by hand category"""
        category_accuracy = {}
        for category, stats in self.model_performance['accuracy_by_category'].items():
            if stats['total'] > 0:
                category_accuracy[category] = stats['correct'] / stats['total']

        return category_accuracy

    def get_model_improvements(self) -> Dict[str, Any]:
        """Analyze historical data to suggest model improvements"""
        improvements = {
            'suggested_adjustments': {},
            'profitable_positions': [],
            'troublesome_categories': []
        }

        # Analyze position profitability
        position_profits = {}
        for hand in self.hand_history[-100:]:  # Last 100 hands
            pos = hand.get('position', 'unknown')
            if pos not in position_profits:
                position_profits[pos] = []
            position_profits[pos].append(hand.get('profit_loss', 0))

        for position, profits in position_profits.items():
            if profits:
                avg_profit = sum(profits) / len(profits)
                if avg_profit > 0:
                    improvements['profitable_positions'].append(position)

        # Identify troublesome categories
        category_accuracy = self.get_category_accuracy()
        for category, accuracy in category_accuracy.items():
            if accuracy < 0.5:  # Less than 50% accuracy
                improvements['troublesome_categories'].append(category)

        return improvements

    def _calculate_outs(self, player_cards: List[Card], community_cards: List[Card]) -> Tuple[int, List[str]]:
        """Calculate number of outs and what you're drawing to"""
        # Simplified outs calculation
        # In reality, this would be much more sophisticated
        outs = 0
        drawing_to = []

        # This is a placeholder implementation
        # Real outs calculation depends on specific hand analysis

        return outs, drawing_to

    def _recommend_action(self, equity: float, pot_odds: float, hand_category: str) -> str:
        """Recommend action based on equity, pot odds, and hand strength"""
        if hand_category in ['monster', 'strong']:
            return 'raise'
        elif hand_category == 'medium':
            if equity > pot_odds + 0.1:
                return 'call'
            else:
                return 'fold'
        elif hand_category == 'weak':
            if equity > pot_odds:
                return 'call'
            else:
                return 'fold'
        else:
            return 'fold'

    def _get_remaining_deck(self, known_cards: List[Card]) -> List[Card]:
        """Get cards remaining in deck"""
        known_ranks = {(card.rank, card.suit) for card in known_cards}
        return [card for card in self.deck if (card.rank, card.suit) not in known_ranks]

    def _deal_random_cards(self, deck: List[Card], count: int) -> List[Card]:
        """Deal random cards from deck"""
        if len(deck) < count:
            return []
        return np.random.choice(deck, size=count, replace=False).tolist()

    def _evaluate_hand_value(self, hole_cards: List[Card], community_cards: List[Card]) -> int:
        """Evaluate hand value for comparison (higher is better)"""
        all_cards = hole_cards + community_cards
        hand = self._convert_cards_for_evaluation(all_cards)

        # Improved hand ranking for comparison with tie-breaking
        if self._is_royal_flush(hand):
            return 10000
        elif self._is_straight_flush(hand):
            high_card = self._get_straight_flush_high_card(hand)
            return 9000 + high_card
        elif self._is_four_of_a_kind(hand):
            return 8000 + self._get_quads_value(hand)
        elif self._is_full_house(hand):
            trips_value = self._get_full_house_trips_value(hand)
            pair_value = self._get_full_house_pair_value(hand)
            return 7000 + (trips_value * 100) + pair_value
        elif self._is_flush(hand):
            return 6000 + self._get_flush_value(hand)
        elif self._is_straight(hand):
            high_card = self._get_straight_high_card(hand)
            return 5000 + high_card
        elif self._is_three_of_a_kind(hand):
            trips_value = self._get_trips_value(hand)
            kickers = self._get_kicker_values(hand, [trips_value])
            return 4000 + (trips_value * 1000) + sum(kickers[:2])
        elif self._is_two_pair(hand):
            pairs = self._get_two_pair_values(hand)
            kicker = self._get_kicker_values(hand, pairs)[0]
            return 3000 + (pairs[0] * 1000) + (pairs[1] * 100) + kicker
        elif self._is_pair(hand):
            pair_value = self._get_pair_value(hand)
            kickers = self._get_kicker_values(hand, [pair_value])
            return 2000 + (pair_value * 1000) + sum(kickers[:3])
        else:
            high_card = self._get_high_card_value(hand)
            kickers = self._get_kicker_values(hand, [high_card])
            return 1000 + (high_card * 1000) + sum(kickers[i] * (100 ** (2-i)) for i in range(4))

    def _get_high_card_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get value of highest card"""
        ranks = [self._rank_to_value(r[0]) for r in hand]
        return max(ranks)

    def _get_quads_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get value for four of a kind"""
        ranks = [r[0] for r in hand]
        for rank in set(ranks):
            if ranks.count(rank) == 4:
                return self._rank_to_value(rank)
        return 0

    def _get_full_house_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get value for full house (three of a kind value)"""
        rank_counts = {}
        for rank in [r[0] for r in hand]:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        trips = max([rank for rank, count in rank_counts.items() if count >= 3])
        return self._rank_to_value(trips)

    def _get_flush_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get value for flush (highest card)"""
        return self._get_high_card_value(hand)

    def _get_straight_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get value for straight (highest card)"""
        ranks = sorted(set([self._rank_to_value(r[0]) for r in hand]))

        # Check for wheel straight
        if set(ranks) == {2, 3, 4, 5, 14}:
            return 5  # 5-high straight

        # Regular straight
        for i in range(len(ranks) - 4):
            if ranks[i+4] - ranks[i] == 4:
                return ranks[i+4]

        return 0

    def _get_trips_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get value for three of a kind"""
        ranks = [r[0] for r in hand]
        for rank in set(ranks):
            if ranks.count(rank) >= 3:
                return self._rank_to_value(rank)
        return 0

    def _get_two_pair_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get value for two pair (higher pair)"""
        rank_counts = {}
        for rank in [r[0] for r in hand]:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        pairs = sorted([self._rank_to_value(rank) for rank, count in rank_counts.items() if count >= 2])
        return pairs[-1] if pairs else 0

    def _get_pair_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get value for pair"""
        ranks = [r[0] for r in hand]
        for rank in set(ranks):
            if ranks.count(rank) >= 2:
                return self._rank_to_value(rank)
        return 0

    def _get_straight_flush_high_card(self, hand: List[Tuple[str, str]]) -> int:
        """Get highest card in straight flush"""
        return self._get_straight_high_card(hand)

    def _get_straight_high_card(self, hand: List[Tuple[str, str]]) -> int:
        """Get highest card in straight"""
        ranks = sorted(set([self._rank_to_value(r[0]) for r in hand]))

        # Check for wheel straight
        if set(ranks) == {2, 3, 4, 5, 14}:
            return 5  # 5-high straight

        # Regular straight
        for i in range(len(ranks) - 4):
            if ranks[i+4] - ranks[i] == 4:
                return ranks[i+4]

        return max(ranks)

    def _get_full_house_trips_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get three of a kind value in full house"""
        rank_counts = {}
        for rank in [r[0] for r in hand]:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        trips = max([rank for rank, count in rank_counts.items() if count >= 3])
        return self._rank_to_value(trips)

    def _get_full_house_pair_value(self, hand: List[Tuple[str, str]]) -> int:
        """Get pair value in full house"""
        rank_counts = {}
        for rank in [r[0] for r in hand]:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        pair = max([rank for rank, count in rank_counts.items() if count >= 2 and rank != max([r for r, c in rank_counts.items() if c >= 3])])
        return self._rank_to_value(pair)

    def _get_kicker_values(self, hand: List[Tuple[str, str]], exclude_ranks: List[str]) -> List[int]:
        """Get kicker values excluding specified ranks"""
        all_ranks = [self._rank_to_value(r[0]) for r in hand]
        kickers = [rank for rank in all_ranks if rank not in exclude_ranks]
        return sorted(kickers, reverse=True)

    def _get_two_pair_values(self, hand: List[Tuple[str, str]]) -> List[int]:
        """Get values of both pairs in two pair hand"""
        rank_counts = {}
        for rank in [r[0] for r in hand]:
            rank_counts[rank] = rank_counts.get(rank, 0) + 1

        pairs = sorted([self._rank_to_value(rank) for rank, count in rank_counts.items() if count >= 2], reverse=True)
        return pairs