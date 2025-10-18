"""
Blackjack odds calculation algorithms
"""
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
import math
from .blackjack_detector import BlackjackCard, BlackjackGameState


@dataclass
class BlackjackOdds:
    """Calculated blackjack odds and probabilities"""
    player_win_probability: float  # Probability of player winning
    dealer_bust_probability: float  # Probability dealer will bust
    player_bust_probability: float  # Probability player will bust if hitting
    surrender_value: float  # Expected value of surrendering
    hit_value: float  # Expected value of hitting
    stand_value: float  # Expected value of standing
    double_value: float  # Expected value of doubling down
    split_value: float  # Expected value of splitting (if applicable)
    recommended_action: str  # 'hit', 'stand', 'double', 'split', 'surrender'
    true_count: float  # Card counting (simplified)
    deck_penetration: float  # How many cards have been played

    # Enhanced fields for comprehensive analysis
    blackjack_probability: float  # Probability of getting blackjack
    push_probability: float  # Probability of tie/push
    face_card_probability: float  # Probability of next card being 10-value
    ace_probability: float  # Probability of next card being ace
    ten_rich_probability: float  # Probability deck is favorable (hi true count)
    double_down_advantage: float  # When double down is advantageous
    split_advantage: float  # When splitting is advantageous
    insurance_value: float  # Expected value of insurance (if offered)
    bankroll_risk: float  # Risk to bankroll for current bet size

    # Advanced strategy fields
    strategy_deviation_applied: bool = False  # Whether count-based deviation was applied
    strategy_deviation_reason: str = ''  # Reason for deviation
    count_impact: float = 0.0  # Impact of count on strategy
    running_count: int = 0  # Current running count
    cards_seen: int = 0  # Total cards seen
    decks_remaining: float = 6.0  # Estimated decks remaining


class AdvancedBlackjackEngine:
    """Advanced blackjack strategy engine with research-based algorithms"""

    def __init__(self):
        self.composition_strategy = self._build_composition_dependent_strategy()
        self.risk_averse_strategy = self._build_risk_averse_strategy()
        self.optimal_bet_sizing = self._build_optimal_bet_sizing()

    def _build_composition_dependent_strategy(self) -> Dict[str, str]:
        """Build composition-dependent strategy (more sophisticated than basic strategy)"""
        # This would be based on research from blackjack experts
        # Simplified version for demonstration
        strategy = {}

        # Adjust basic strategy based on card composition
        for player_score in range(5, 22):
            for dealer_card in range(2, 12):
                base_action = self.get_optimal_strategy(player_score, dealer_card)

                # Adjust for composition awareness
                if player_score == 16 and dealer_card in [9, 10, 1]:
                    strategy[f"{player_score}_{dealer_card}"] = 'stand'  # Composition dependent
                else:
                    strategy[f"{player_score}_{dealer_card}"] = base_action

        return strategy

    def _build_risk_averse_strategy(self) -> Dict[str, str]:
        """Build risk-averse strategy for conservative play"""
        strategy = {}

        for player_score in range(5, 22):
            for dealer_card in range(2, 12):
                action = self.get_optimal_strategy(player_score, dealer_card)

                # Make more conservative adjustments
                if action == 'hit' and player_score >= 16:
                    strategy[f"{player_score}_{dealer_card}"] = 'stand'
                else:
                    strategy[f"{player_score}_{dealer_card}"] = action

        return strategy

    def _build_optimal_bet_sizing(self) -> Dict[float, float]:
        """Build optimal bet sizing based on true count"""
        # Kelly Criterion for blackjack betting
        bet_sizing = {}

        for true_count in range(-5, 16):
            if true_count <= 0:
                bet_sizing[true_count] = 1.0  # Minimum bet
            else:
                # Kelly-optimal bet sizing
                advantage = true_count * 0.005  # 0.5% per count
                bet_sizing[true_count] = min(advantage * 10, 5.0)  # Cap at 5x bet

        return bet_sizing

    def get_composition_strategy(self, player_score: int, dealer_upcard: int) -> str:
        """Get composition-dependent strategy"""
        key = f"{player_score}_{dealer_upcard}"
        return self.composition_strategy.get(key, 'stand')

    def get_risk_averse_action(self, player_score: int, dealer_upcard: int) -> str:
        """Get risk-averse action"""
        key = f"{player_score}_{dealer_upcard}"
        return self.risk_averse_strategy.get(key, 'stand')

    def get_optimal_bet_size(self, true_count: float, base_bet: float = 10.0) -> float:
        """Get optimal bet size based on true count"""
        # Find closest true count in our table
        closest_count = min(self.optimal_bet_sizing.keys(),
                          key=lambda x: abs(x - true_count))

        multiplier = self.optimal_bet_sizing[closest_count]
        return base_bet * multiplier


class BlackjackOddsCalculator:
    """Calculates blackjack odds and probabilities"""

    def __init__(self, game_rules=None):
        # Use provided game rules or create defaults
        self.game_rules = game_rules

        # Standard deck composition for counting (Hi-Lo system)
        self.card_values = {
            '2': 1, '3': 1, '4': 1, '5': 1, '6': 1,
            '7': 0, '8': 0, '9': 0, '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
        }

        # Advanced card counting systems
        self.counting_systems = {
            'hi_lo': self.card_values,
            'zen_count': {
                '2': 1, '3': 2, '4': 2, '5': 2, '6': 2, '7': 1,
                '8': 0, '9': 0, '10': -2, 'J': -2, 'Q': -2, 'K': -2, 'A': -1
            },
            'wong_halves': {
                '2': 0.5, '3': 1, '4': 1, '5': 1.5, '6': 1, '7': 0.5,
                '8': 0, '9': -0.5, '10': -1, 'J': -1, 'Q': -1, 'K': -1, 'A': -1
            }
        }

        # Basic strategy tables based on game rules
        self.basic_strategy = self._create_basic_strategy()

        # Dealer probabilities (probability dealer has each card)
        self.dealer_probabilities = self._calculate_dealer_probabilities()

        # Card counting state
        self.running_count = 0
        self.true_count = 0
        self.cards_seen = 0
        self.decks_remaining = self.game_rules.decks if self.game_rules else 6

    def calculate_odds(self, game_state: BlackjackGameState) -> Optional[BlackjackOdds]:
        """Calculate comprehensive blackjack odds with MDP-aware state modeling"""
        try:
            if not game_state.player_cards or not game_state.dealer_upcard:
                return None

            player_score = game_state.player_score
            dealer_upcard = game_state.dealer_upcard.value

            # Enhanced state representation for MDP (expert analysis)
            current_state = self._create_mdp_state(player_score, dealer_upcard,
                                                 game_state.player_cards, self.true_count)

            # Update card count with current hand (including perception uncertainty)
            all_cards = game_state.player_cards + game_state.dealer_cards
            self.update_count(all_cards)

            # Apply perception uncertainty factor (from expert analysis)
            uncertainty_factor = 1 - 0.02  # 2% base uncertainty

            # Get strategy with deviations based on true count
            strategy_info = self.get_strategy_with_deviations(
                player_score, dealer_upcard, self.true_count,
                game_state.player_can_double, game_state.player_can_split
            )

            # Calculate probabilities (adjusted for true count)
            player_win_prob = self._calculate_player_win_probability(
                player_score, dealer_upcard, game_state.deck_count
            )
            dealer_bust_prob = self._calculate_dealer_bust_probability(dealer_upcard)

            # Adjust probabilities based on true count
            if self.true_count > 0:
                player_win_prob = min(0.95, player_win_prob + (self.true_count * 0.005))
            elif self.true_count < 0:
                player_win_prob = max(0.05, player_win_prob + (self.true_count * 0.005))

            # Calculate expected values for different actions
            hit_value = self._calculate_hit_value(player_score, dealer_upcard)
            stand_value = self._calculate_stand_value(player_score, dealer_upcard)
            double_value = self._calculate_double_value(player_score, dealer_upcard)
            split_value = self._calculate_split_value(game_state.player_cards, dealer_upcard)

            # Determine recommended action (use deviation strategy if applicable)
            if strategy_info['deviation_applied']:
                recommended_action = strategy_info['base_action']
            else:
                recommended_action = self._get_recommended_action(
                    player_score, dealer_upcard, game_state, hit_value, stand_value, double_value, split_value
                )

            # Calculate card counting info
            true_count = self.true_count
            deck_penetration = self.cards_seen / ((self.game_rules.decks if self.game_rules else 6) * 52)

            # Calculate comprehensive probabilities
            blackjack_prob = self._calculate_blackjack_probability(game_state.player_cards)
            push_prob = self._calculate_push_probability(player_score, dealer_upcard)
            face_card_prob = self._calculate_face_card_probability(game_state.true_count)
            ace_prob = self._calculate_ace_probability(game_state.true_count)
            ten_rich_prob = self._calculate_ten_rich_probability(game_state.true_count)
            double_advantage = self._calculate_double_advantage(player_score, dealer_upcard)
            split_advantage = self._calculate_split_advantage(game_state.player_cards, dealer_upcard)
            insurance_value = self._calculate_insurance_value(dealer_upcard, game_state.true_count)
            bankroll_risk = self._calculate_bankroll_risk(game_state)

            return BlackjackOdds(
                player_win_probability=player_win_prob,
                dealer_bust_probability=dealer_bust_prob,
                player_bust_probability=self._calculate_player_bust_probability(player_score),
                surrender_value=self._calculate_surrender_value(player_score, dealer_upcard),
                hit_value=hit_value,
                stand_value=stand_value,
                double_value=double_value,
                split_value=split_value,
                recommended_action=recommended_action,
                true_count=true_count,
                deck_penetration=deck_penetration,
                blackjack_probability=blackjack_prob,
                push_probability=push_prob,
                face_card_probability=face_card_prob,
                ace_probability=ace_prob,
                ten_rich_probability=ten_rich_prob,
                double_down_advantage=double_advantage,
                split_advantage=split_advantage,
                insurance_value=insurance_value,
                bankroll_risk=bankroll_risk,

                # Add new fields for advanced strategy
                strategy_deviation_applied=strategy_info['deviation_applied'],
                strategy_deviation_reason=strategy_info['deviation_reason'],
                count_impact=strategy_info['count_impact'],
                running_count=self.running_count,
                cards_seen=self.cards_seen,
                decks_remaining=self.decks_remaining
            )

        except Exception as e:
            print(f"Error calculating blackjack odds: {e}")
            return None

    def _calculate_player_win_probability(self, player_score: int, dealer_upcard: int, decks: int) -> float:
        """Calculate probability of player winning"""
        if player_score == 21:
            # Blackjack - very high probability of winning
            return 0.85 if dealer_upcard != 21 else 0.5

        if player_score > 21:
            return 0.0  # Already busted

        # Simplified probability calculation
        # In reality, this would use more sophisticated Monte Carlo simulation
        base_prob = 0.4

        # Adjust based on player score and dealer upcard
        if player_score >= 17:
            base_prob += 0.1
        elif player_score <= 11:
            base_prob += 0.15

        if dealer_upcard >= 7:
            base_prob -= 0.1
        elif dealer_upcard <= 3:
            base_prob += 0.1

        return max(0.0, min(1.0, base_prob))

    def _calculate_dealer_bust_probability(self, dealer_upcard: int) -> float:
        """Calculate probability dealer will bust"""
        # Dealer must hit on 16, stand on 17+
        # Probability depends on dealer's upcard

        bust_probabilities = {
            2: 0.35, 3: 0.37, 4: 0.40, 5: 0.42, 6: 0.42,
            7: 0.26, 8: 0.24, 9: 0.23, 10: 0.21, 1: 0.35  # 1=Ace
        }

        return bust_probabilities.get(dealer_upcard, 0.3)

    def _calculate_player_bust_probability(self, player_score: int) -> float:
        """Calculate probability player will bust if hitting"""
        if player_score >= 21:
            return 1.0 if player_score > 21 else 0.0

        # Probability of busting depends on current score
        bust_probabilities = {
            11: 0.0, 12: 0.31, 13: 0.39, 14: 0.56, 15: 0.58,
            16: 0.62, 17: 0.69, 18: 0.77, 19: 0.85, 20: 0.92
        }

        return bust_probabilities.get(player_score, 0.5)

    def _calculate_hit_value(self, player_score: int, dealer_upcard: int) -> float:
        """Calculate expected value of hitting"""
        if player_score >= 21:
            return -1.0 if player_score > 21 else 1.0

        win_prob = self._calculate_player_win_probability(player_score + 1, dealer_upcard, 6)
        bust_prob = self._calculate_player_bust_probability(player_score)

        # Expected value: win * 1 + lose * (-1) + tie * 0
        return (win_prob * 1) + ((1 - win_prob - bust_prob) * (-1)) + (bust_prob * (-1))

    def _calculate_stand_value(self, player_score: int, dealer_upcard: int) -> float:
        """Calculate expected value of standing"""
        win_prob = self._calculate_player_win_probability(player_score, dealer_upcard, 6)
        bust_prob = 0.0  # Player doesn't bust when standing

        return (win_prob * 1) + ((1 - win_prob) * (-1))

    def _calculate_double_value(self, player_score: int, dealer_upcard: int) -> float:
        """Calculate expected value of doubling down"""
        if player_score > 11:
            return self._calculate_stand_value(player_score, dealer_upcard)  # Can't double

        # Doubling gives one more card and doubles the bet
        hit_value = self._calculate_hit_value(player_score, dealer_upcard)
        return hit_value * 2  # Double the bet, so double the expected value

    def _calculate_split_value(self, player_cards: List[BlackjackCard], dealer_upcard: int) -> float:
        """Calculate expected value of splitting"""
        if len(player_cards) != 2 or player_cards[0].rank != player_cards[1].rank:
            return -2.0  # Can't split, very negative value

        # Simplified split value
        card_value = player_cards[0].value
        split_values = {
            1: -0.5, 2: -0.3, 3: -0.2, 4: -0.1, 5: -0.1,
            6: 0.0, 7: 0.1, 8: 0.2, 9: 0.3, 10: 0.4
        }

        return split_values.get(card_value, -0.2)

    def _calculate_surrender_value(self, player_score: int, dealer_upcard: int) -> float:
        """Calculate expected value of surrendering"""
        # Surrender gives back half the bet
        stand_value = self._calculate_stand_value(player_score, dealer_upcard)
        return stand_value * 0.5  # Get half back regardless of outcome

    def _get_recommended_action(self, player_score: int, dealer_upcard: int,
                              game_state: BlackjackGameState, hit_value: float,
                              stand_value: float, double_value: float, split_value: float) -> str:
        """Get recommended action using basic strategy"""

        # Check if can split
        if (game_state.player_can_split and
            len(game_state.player_cards) == 2 and
            game_state.player_cards[0].rank == game_state.player_cards[1].rank):

            if split_value > stand_value:
                return 'split'

        # Check if can double
        if (game_state.player_can_double and
            len(game_state.player_cards) == 2):

            if double_value > max(hit_value, stand_value):
                return 'double'

        # Hit or stand decision
        if hit_value > stand_value:
            return 'hit'
        else:
            return 'stand'

    def update_count(self, cards: List[BlackjackCard]) -> None:
        """Update running count with newly seen cards"""
        for card in cards:
            card_value = self.counting_systems[self.game_rules.counting_system if self.game_rules else 'hi_lo'].get(card.rank, 0)
            self.running_count += card_value
            self.cards_seen += 1

        # Update true count
        self.true_count = self.calculate_true_count()

    def calculate_true_count(self) -> float:
        """Calculate true count based on cards seen and remaining decks"""
        if self.cards_seen == 0:
            return 0.0

        # Calculate deck penetration
        total_cards = (self.game_rules.decks if self.game_rules else 6) * 52
        penetration = self.cards_seen / total_cards

        # Estimate remaining decks
        self.decks_remaining = max(1.0, (self.game_rules.decks if self.game_rules else 6) * (1 - penetration))

        # True count = running count / remaining decks
        if self.decks_remaining > 0:
            return self.running_count / self.decks_remaining
        return 0.0

    def get_count_status(self) -> Dict[str, float]:
        """Get current card counting status"""
        return {
            'running_count': self.running_count,
            'true_count': self.true_count,
            'cards_seen': self.cards_seen,
            'decks_remaining': self.decks_remaining,
            'deck_penetration': self.cards_seen / ((self.game_rules.decks if self.game_rules else 6) * 52)
        }

    def reset_count(self) -> None:
        """Reset card counting state (for new shoe)"""
        self.running_count = 0
        self.true_count = 0
        self.cards_seen = 0
        self.decks_remaining = self.game_rules.decks if self.game_rules else 6

    def _calculate_true_count(self, game_state: BlackjackGameState) -> float:
        """Calculate true count for card counting (legacy compatibility)"""
        if not game_state.player_cards or not game_state.dealer_cards:
            return self.true_count

        # Update count with current hand cards
        all_cards = game_state.player_cards + game_state.dealer_cards
        self.update_count(all_cards)

        return self.true_count

    def _calculate_deck_penetration(self, game_state: BlackjackGameState) -> float:
        """Calculate what percentage of the deck has been played"""
        # Simplified calculation
        cards_played = len(game_state.player_cards) + len(game_state.dealer_cards)
        total_cards = game_state.deck_count * 52

        return min(1.0, cards_played / total_cards)

    def _create_basic_strategy(self) -> Dict[Tuple[int, int], str]:
        """Create comprehensive basic strategy lookup table based on game rules"""
        strategy = {}

        # Get rules for strategy variations
        h17 = self.game_rules.dealer_hits_soft_17 if hasattr(self, 'game_rules') else True
        das = self.game_rules.double_after_split if hasattr(self, 'game_rules') else True
        surrender = self.game_rules.surrender_allowed if hasattr(self, 'game_rules') else True

        # Hard totals strategy (H17 rules)
        hard_strategy_h17 = {
            # Player 8 or less: always hit
            8: {2: 'H', 3: 'H', 4: 'H', 5: 'H', 6: 'H', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # Player 9
            9: {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # Player 10
            10: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 11: 'H'},
            # Player 11
            11: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'D', 11: 'H'},
            # Player 12
            12: {2: 'H', 3: 'H', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # Player 13
            13: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # Player 14
            14: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # Player 15
            15: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # Player 16
            16: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # Player 17+
            17: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
            18: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
            19: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
            20: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
            21: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'}
        }

        # S17 adjustments (dealer stands on soft 17)
        if not h17:
            hard_strategy_h17[15][11] = 'H'  # 15 vs A: hit for S17
            hard_strategy_h17[16][11] = 'H'  # 16 vs A: hit for S17

        # Apply hard strategy
        for player_score, dealer_actions in hard_strategy_h17.items():
            for dealer_card, action in dealer_actions.items():
                strategy[(player_score, dealer_card)] = action

        # Soft totals strategy
        soft_strategy = {
            # A,2
            13: {2: 'H', 3: 'H', 4: 'H', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # A,3
            14: {2: 'H', 3: 'H', 4: 'H', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # A,4
            15: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # A,5
            16: {2: 'H', 3: 'H', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # A,6
            17: {2: 'H', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # A,7
            18: {2: 'S', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'S', 8: 'S', 9: 'H', 10: 'S', 11: 'S'},
            # A,8
            19: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
            # A,9
            20: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'},
            # A,10
            21: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'}
        }

        # Apply soft strategy
        for player_score, dealer_actions in soft_strategy.items():
            for dealer_card, action in dealer_actions.items():
                strategy[(player_score, dealer_card)] = action

        # Pair splitting strategy
        pair_strategy = {
            # A,A
            12: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
            # 2,2
            4: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # 3,3
            6: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # 4,4
            8: {2: 'H', 3: 'H', 4: 'H', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # 5,5
            10: {2: 'D', 3: 'D', 4: 'D', 5: 'D', 6: 'D', 7: 'D', 8: 'D', 9: 'D', 10: 'H', 11: 'H'},
            # 6,6
            12: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'H', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # 7,7
            14: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'H', 9: 'H', 10: 'H', 11: 'H'},
            # 8,8
            16: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'P', 8: 'P', 9: 'P', 10: 'P', 11: 'P'},
            # 9,9
            18: {2: 'P', 3: 'P', 4: 'P', 5: 'P', 6: 'P', 7: 'S', 8: 'P', 9: 'P', 10: 'S', 11: 'S'},
            # 10,10
            20: {2: 'S', 3: 'S', 4: 'S', 5: 'S', 6: 'S', 7: 'S', 8: 'S', 9: 'S', 10: 'S', 11: 'S'}
        }

        # Apply pair strategy
        for player_score, dealer_actions in pair_strategy.items():
            for dealer_card, action in dealer_actions.items():
                strategy[(player_score, dealer_card)] = action

        # Surrender strategy (if allowed)
        if surrender:
            surrender_strategy = {
                # Hard 16 vs 9,10,A (except 8,8)
                16: {9: 'R', 10: 'R', 11: 'R'},
                # Hard 15 vs 10
                15: {10: 'R'},
                # Hard 17 vs 11 (S17 only)
                17: {11: 'R'} if not h17 else {}
            }

            for player_score, dealer_actions in surrender_strategy.items():
                for dealer_card, action in dealer_actions.items():
                    strategy[(player_score, dealer_card)] = action

        return strategy

    def _calculate_dealer_probabilities(self) -> Dict[int, float]:
        """Calculate probability distribution of dealer cards"""
        # Each card 2-10 has equal probability, face cards grouped as 10
        probabilities = {}
        for card in range(2, 10):
            probabilities[card] = 4 / 52  # 4 suits per rank

        probabilities[10] = 16 / 52  # 10, J, Q, K (4 each)
        probabilities[1] = 4 / 52    # Aces

        return probabilities

    def _calculate_blackjack_probability(self, player_cards: List[BlackjackCard]) -> float:
        """Calculate probability of having/getting blackjack"""
        if len(player_cards) == 2:
            # Check if current hand is blackjack
            has_ace = any(card.rank == 'A' for card in player_cards)
            has_ten = any(card.value == 10 for card in player_cards)
            if has_ace and has_ten:
                return 1.0  # Already has blackjack

        # Probability of getting blackjack with next card
        # Approximately 4.8% in single deck (adjusted for cards played)
        base_probability = 4.8 / 100

        # Adjust based on cards already seen (simplified)
        return base_probability

    def _calculate_push_probability(self, player_score: int, dealer_upcard: int) -> float:
        """Calculate probability of tie/push"""
        if player_score > 21:
            return 0.0  # Can't push if busted

        # Simplified push probability based on dealer upcard and player score
        push_probabilities = {
            17: 0.15, 18: 0.12, 19: 0.10, 20: 0.08, 21: 0.05
        }

        return push_probabilities.get(player_score, 0.1)

    def _calculate_face_card_probability(self, true_count: float) -> float:
        """Calculate probability of next card being a face card (10-value)"""
        # Base probability of face card in single deck: 16/52 = 30.8%
        base_prob = 16 / 52

        # Adjust based on true count (positive count means more face cards left)
        count_adjustment = true_count * 0.02  # 2% adjustment per count point

        return min(0.5, max(0.1, base_prob + count_adjustment))

    def _calculate_ace_probability(self, true_count: float) -> float:
        """Calculate probability of next card being an ace"""
        # Base probability of ace in single deck: 4/52 ≈ 7.7%
        base_prob = 4 / 52

        # Negative count means fewer aces left (aces count as -1 in Hi-Lo)
        count_adjustment = -true_count * 0.01  # 1% adjustment per count point

        return min(0.3, max(0.01, base_prob + count_adjustment))

    def _calculate_ten_rich_probability(self, true_count: float) -> float:
        """Calculate probability that deck favors player (high 10-value cards)"""
        return max(0.0, min(1.0, 0.5 + (true_count * 0.1)))

    def _calculate_double_advantage(self, player_score: int, dealer_upcard: int) -> float:
        """Calculate when double down is advantageous"""
        advantage = 0.0

        # Double down is advantageous when player has 9, 10, or 11
        if player_score in [9, 10, 11]:
            # More advantageous against dealer's weak upcards
            if dealer_upcard in [2, 3, 4, 5, 6]:
                advantage = 0.8  # Strong advantage
            elif dealer_upcard in [7, 8, 9]:
                advantage = 0.4  # Moderate advantage
            else:
                advantage = 0.2  # Slight advantage

        return advantage

    def _calculate_split_advantage(self, player_cards: List[BlackjackCard], dealer_upcard: int) -> float:
        """Calculate when splitting is advantageous"""
        if len(player_cards) != 2 or player_cards[0].rank != player_cards[1].rank:
            return 0.0

        card_value = player_cards[0].value
        advantage = 0.0

        # Splitting aces is always advantageous
        if card_value == 11:  # Aces
            advantage = 0.9
        elif card_value == 8:  # 8s
            advantage = 0.7
        elif card_value == 9:  # 9s
            if dealer_upcard in [2, 3, 4, 5, 6, 8, 9]:
                advantage = 0.6
        elif card_value in [2, 3, 7]:  # Generally good to split
            if dealer_upcard in [2, 3, 4, 5, 6, 7]:
                advantage = 0.4

        return advantage

    def _calculate_insurance_value(self, dealer_upcard: int, true_count: float) -> float:
        """Calculate expected value of insurance bet"""
        if dealer_upcard != 11:  # Only when dealer shows ace
            return -1.0  # Insurance not offered

        # Insurance pays 2:1 if dealer has blackjack
        # Probability dealer has blackjack given showing ace
        blackjack_prob = self._dealer_blackjack_probability(true_count)

        # Expected value = (blackjack_prob * 2) + ((1 - blackjack_prob) * -1)
        return (blackjack_prob * 2) + ((1 - blackjack_prob) * -1)

    def _dealer_blackjack_probability(self, true_count: float) -> float:
        """Calculate probability dealer has blackjack when showing ace"""
        # Base probability: 4/13 (other 3 aces and 16 ten-value cards)
        base_prob = 4 / 13

        # Adjust based on true count (negative count means fewer ten-value cards)
        adjustment = -true_count * 0.02

        return max(0.05, min(0.5, base_prob + adjustment))

    def _calculate_bankroll_risk(self, game_state: BlackjackGameState) -> float:
        """Calculate risk to bankroll for current bet size"""
        if not game_state.current_bet or not game_state.player_bankroll:
            return 0.0

        # Risk is bet size as percentage of bankroll
        risk_percentage = game_state.current_bet / game_state.player_bankroll

        # High risk if betting more than 5% of bankroll
        if risk_percentage > 0.05:
            return 0.8  # High risk
        elif risk_percentage > 0.02:
            return 0.5  # Medium risk
        else:
            return 0.2  # Low risk

    def get_optimal_strategy(self, player_score: int, dealer_upcard: int, can_double: bool = True) -> str:
        """Get optimal play according to basic strategy"""
        key = (player_score, dealer_upcard)
        return self.basic_strategy.get(key, 'stand')

    def get_strategy_with_deviations(self, player_score: int, dealer_upcard: int,
                                   true_count: float, can_double: bool = True,
                                   can_split: bool = True) -> Dict[str, Any]:
        """
        Get strategy recommendation including deviations based on true count

        Based on research-based playing deviations for advantage play
        """
        # Get base strategy
        base_strategy = self.get_optimal_strategy(player_score, dealer_upcard, can_double)
        strategy_info = {
            'base_action': base_strategy,
            'deviation_applied': False,
            'deviation_reason': '',
            'count_impact': 0.0
        }

        # Insurance deviations (when dealer shows Ace)
        if dealer_upcard == 11:  # Ace
            if true_count >= 3.0:
                strategy_info.update({
                    'base_action': 'insurance',
                    'deviation_applied': True,
                    'deviation_reason': 'High true count makes insurance profitable',
                    'count_impact': true_count * 0.02  # 2% per count
                })

        # 16 vs 10 deviation
        elif player_score == 16 and dealer_upcard == 10:
            if true_count >= 0:
                strategy_info.update({
                    'base_action': 'stand',
                    'deviation_applied': True,
                    'deviation_reason': 'Positive count favors standing on 16 vs 10',
                    'count_impact': true_count * 0.01
                })

        # 15 vs 10 deviation
        elif player_score == 15 and dealer_upcard == 10:
            if true_count >= 4:
                strategy_info.update({
                    'base_action': 'stand',
                    'deviation_applied': True,
                    'deviation_reason': 'Very high count makes standing on 15 vs 10 correct',
                    'count_impact': true_count * 0.015
                })

        # 12 vs 4 deviation (stand instead of hit)
        elif player_score == 12 and dealer_upcard == 4:
            if true_count >= 2:
                strategy_info.update({
                    'base_action': 'stand',
                    'deviation_applied': True,
                    'deviation_reason': 'Positive count favors standing on 12 vs 4',
                    'count_impact': true_count * 0.01
                })

        # 12 vs 5 deviation
        elif player_score == 12 and dealer_upcard == 5:
            if true_count >= 1:
                strategy_info.update({
                    'base_action': 'stand',
                    'deviation_applied': True,
                    'deviation_reason': 'Positive count favors standing on 12 vs 5',
                    'count_impact': true_count * 0.01
                })

        # 12 vs 6 deviation
        elif player_score == 12 and dealer_upcard == 6:
            if true_count >= 0:
                strategy_info.update({
                    'base_action': 'stand',
                    'deviation_applied': True,
                    'deviation_reason': 'Neutral/positive count favors standing on 12 vs 6',
                    'count_impact': true_count * 0.01
                })

        # 13 vs 2 deviation (hit instead of stand)
        elif player_score == 13 and dealer_upcard == 2:
            if true_count <= -1:
                strategy_info.update({
                    'base_action': 'hit',
                    'deviation_applied': True,
                    'deviation_reason': 'Negative count favors hitting 13 vs 2',
                    'count_impact': abs(true_count) * 0.01
                })

        # 10 double vs 10 (double instead of hit)
        elif player_score == 10 and dealer_upcard == 10 and can_double:
            if true_count >= 4:
                strategy_info.update({
                    'base_action': 'double',
                    'deviation_applied': True,
                    'deviation_reason': 'High count makes doubling 10 vs 10 profitable',
                    'count_impact': true_count * 0.02
                })

        # 9 double vs 7 (double instead of hit)
        elif player_score == 9 and dealer_upcard == 7 and can_double:
            if true_count >= 3:
                strategy_info.update({
                    'base_action': 'double',
                    'deviation_applied': True,
                    'deviation_reason': 'High count makes doubling 9 vs 7 profitable',
                    'count_impact': true_count * 0.015
                })

        # 13 vs 3 deviation (stand instead of hit)
        elif player_score == 13 and dealer_upcard == 3:
            if true_count >= 2:
                strategy_info.update({
                    'base_action': 'stand',
                    'deviation_applied': True,
                    'deviation_reason': 'Positive count favors standing on 13 vs 3',
                    'count_impact': true_count * 0.01
                })

        return strategy_info

    def _create_mdp_state(self, player_score: int, dealer_upcard: int,
                         player_cards: List[BlackjackCard], true_count: float) -> Dict[str, Any]:
        """Create MDP state representation (expert analysis)"""
        # Enhanced state tuple for better strategy modeling
        has_usable_ace = any(card.rank == 'A' and player_score <= 21 for card in player_cards)
        deck_composition_factor = self._categorize_true_count(true_count)

        state = {
            'player_total': player_score,
            'dealer_upcard': dealer_upcard,
            'has_usable_ace': has_usable_ace,
            'true_count': true_count,
            'deck_composition': deck_composition_factor,
            'state_tuple': (player_score, dealer_upcard, has_usable_ace, deck_composition_factor)
        }

        return state

    def _categorize_true_count(self, true_count: float) -> str:
        """Categorize true count for state representation"""
        if true_count > 2:
            return 'very_high'
        elif true_count > 0:
            return 'high'
        elif true_count > -2:
            return 'neutral'
        else:
            return 'low'

    def get_advanced_strategy(self, player_score: int, dealer_upcard: int,
                             true_count: float, can_double: bool = True,
                             can_split: bool = False) -> Dict[str, Any]:
        """Get advanced strategy considering card counting"""
        base_strategy = self.get_optimal_strategy(player_score, dealer_upcard, can_double)

        # Adjust strategy based on true count
        strategy_adjustments = {
            'action': base_strategy,
            'deviation_reason': '',
            'count_impact': 0.0
        }

        # Positive count favors player
        if true_count > 1:
            if base_strategy == 'stand' and player_score >= 15:
                strategy_adjustments['action'] = 'stand'  # More conservative
                strategy_adjustments['deviation_reason'] = 'Positive count - protect good hand'
                strategy_adjustments['count_impact'] = 0.2
        elif true_count < -1:
            if base_strategy == 'hit' and player_score <= 16:
                strategy_adjustments['action'] = 'hit'  # More aggressive hitting
                strategy_adjustments['deviation_reason'] = 'Negative count - hit more'
                strategy_adjustments['count_impact'] = -0.1

        return strategy_adjustments


class BlackjackModelTracker:
    """Tracks blackjack model performance and improves recommendations over time"""

    def __init__(self):
        self.history_file = 'blackjack_model_history.json'
        self.hand_history = []
        self.model_performance = {
            'total_hands': 0,
            'correct_predictions': 0,
            'accuracy_by_score': {},
            'profit_loss_tracking': [],
            'true_count_accuracy': []
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
            logger.warning(f"Failed to save blackjack history: {e}")

    def record_hand_result(self, game_state: BlackjackGameState, predicted_action: str,
                          actual_result: str, profit_loss: float) -> None:
        """Record the result of a blackjack hand for model improvement"""
        hand_record = {
            'timestamp': time.time(),
            'player_score': game_state.player_score,
            'dealer_upcard': game_state.dealer_upcard.value if game_state.dealer_upcard else 0,
            'true_count': game_state.true_count,
            'predicted_action': predicted_action,
            'actual_result': actual_result,
            'profit_loss': profit_loss,
            'current_bet': game_state.current_bet or 0
        }

        self.hand_history.append(hand_record)
        self.model_performance['total_hands'] += 1

        # Update accuracy tracking
        if predicted_action == actual_result:
            self.model_performance['correct_predictions'] += 1

        # Track profit/loss
        self.model_performance['profit_loss_tracking'].append(profit_loss)

        # Track accuracy by player score
        score_range = self._categorize_score(game_state.player_score)
        if score_range not in self.model_performance['accuracy_by_score']:
            self.model_performance['accuracy_by_score'][score_range] = {'correct': 0, 'total': 0}

        self.model_performance['accuracy_by_score'][score_range]['total'] += 1
        if predicted_action == actual_result:
            self.model_performance['accuracy_by_score'][score_range]['correct'] += 1

        # Track true count accuracy (for card counting)
        if game_state.true_count != 0:
            self.model_performance['true_count_accuracy'].append(abs(game_state.true_count))

        # Keep only recent history (last 500 hands)
        if len(self.hand_history) > 500:
            self.hand_history = self.hand_history[-500:]

        self._save_history()

    def _categorize_score(self, score: int) -> str:
        """Categorize hand score for tracking"""
        if score <= 11:
            return 'low'
        elif score <= 16:
            return 'medium'
        elif score <= 20:
            return 'high'
        else:
            return 'blackjack'

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

    def get_true_count_effectiveness(self) -> float:
        """Get effectiveness of true count predictions"""
        if not self.model_performance['true_count_accuracy']:
            return 0.0

        # Lower average error means better effectiveness
        avg_error = sum(self.model_performance['true_count_accuracy']) / len(self.model_performance['true_count_accuracy'])
        return max(0.0, 1.0 - (avg_error / 5.0))  # Normalize to 0-1 scale

    def get_model_improvements(self) -> Dict[str, Any]:
        """Analyze historical data to suggest model improvements"""
        improvements = {
            'troublesome_scores': [],
            'profitable_true_counts': [],
            'suggested_adjustments': {}
        }

        # Analyze score accuracy
        for score_range, stats in self.model_performance['accuracy_by_score'].items():
            if stats['total'] > 10:  # Minimum sample size
                accuracy = stats['correct'] / stats['total']
                if accuracy < 0.6:  # Less than 60% accuracy
                    improvements['troublesome_scores'].append(score_range)

        # Analyze true count profitability
        count_profits = {}
        for hand in self.hand_history[-100:]:  # Last 100 hands
            true_count = hand.get('true_count', 0)
            count_range = self._categorize_true_count(true_count)
            if count_range not in count_profits:
                count_profits[count_range] = []
            count_profits[count_range].append(hand.get('profit_loss', 0))

        for count_range, profits in count_profits.items():
            if profits:
                avg_profit = sum(profits) / len(profits)
                if avg_profit > 0.1:  # Profitable threshold
                    improvements['profitable_true_counts'].append(count_range)

        return improvements

    def _categorize_true_count(self, true_count: float) -> str:
        """Categorize true count for analysis"""
        if true_count > 2:
            return 'high_positive'
        elif true_count > 0:
            return 'low_positive'
        elif true_count > -2:
            return 'neutral'
        else:
            return 'negative'