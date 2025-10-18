"""
Blackjack game state detection module
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import config.config as config


@dataclass
class BlackjackCard:
    """Represents a blackjack card"""
    rank: str  # 'A', '2'-'10', 'J', 'Q', 'K'
    suit: str  # 'hearts', 'diamonds', 'clubs', 'spades'
    value: int  # Blackjack value (A=1 or 11, face=10)
    confidence: float

    def __post_init__(self):
        """Calculate blackjack value after initialization"""
        if self.rank in ['J', 'Q', 'K']:
            self.value = 10
        elif self.rank == 'A':
            self.value = 11  # Will be adjusted to 1 if needed
        else:
            self.value = int(self.rank)


@dataclass
class BlackjackGameState:
    """Current state of a blackjack game"""
    player_cards: List[BlackjackCard]
    dealer_cards: List[BlackjackCard]
    dealer_upcard: Optional[BlackjackCard]  # The visible dealer card
    player_score: int
    dealer_score: int  # Only includes visible cards
    game_phase: str  # 'betting', 'playing', 'dealer_turn', 'finished'
    player_can_hit: bool = True
    player_can_stand: bool = True
    player_can_double: bool = True
    player_can_split: bool = False
    deck_count: int = 6  # Number of decks (assumed)
    current_bet: Optional[int] = None  # Current bet amount
    total_bet: Optional[int] = None    # Total amount bet this hand
    player_bankroll: Optional[int] = None  # Player's remaining bankroll
    true_count: float = 0.0  # Card counting true count
    running_count: int = 0   # Card counting running count
    cards_played: int = 0    # Number of cards played for penetration
    confidence: float = 0.0


class BlackjackCardDetector:
    """Detects and recognizes blackjack cards from images"""

    def __init__(self, config: config.GameConfig):
        self.config = config
        self.card_templates = {}
        self._load_card_templates()

    def _load_card_templates(self) -> None:
        """Load card template images for recognition"""
        # Placeholder - would load actual card templates in real implementation
        self.card_templates = {
            'ranks': ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'],
            'suits': ['hearts', 'diamonds', 'clubs', 'spades']
        }

    def detect_card(self, card_image: np.ndarray) -> Optional[BlackjackCard]:
        """Detect a single blackjack card from an image region"""
        try:
            # Preprocess the card image
            processed = self._preprocess_card(card_image)

            # Extract features for rank and suit detection
            rank, rank_conf = self._detect_rank(processed)
            suit, suit_conf = self._detect_suit(processed)

            if rank and suit:
                confidence = min(rank_conf, suit_conf)
                card = BlackjackCard(rank=rank, suit=suit, value=0, confidence=confidence)
                return card

        except Exception as e:
            print(f"Error detecting blackjack card: {e}")

        return None

    def _preprocess_card(self, image: np.ndarray) -> np.ndarray:
        """Preprocess card image for better recognition"""
        # Similar to poker card preprocessing
        height, width = image.shape[:2]
        target_size = 100
        if height != target_size or width != target_size:
            image = cv2.resize(image, (target_size, target_size))

        # Enhance contrast
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)

        # Convert to grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return gray

    def _detect_rank(self, processed: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect card rank from processed image using OCR and feature detection"""
        # Try OCR first (most reliable for blackjack)
        ocr_result = self._ocr_rank_detection(processed)
        if ocr_result and ocr_result[1] > 0.7:
            return ocr_result

        # Try feature-based detection
        feature_result = self._feature_based_rank(processed)
        if feature_result and feature_result[1] > 0.5:
            return feature_result

        # Fallback
        return self._pattern_rank_detection(processed)

    def _detect_suit(self, processed: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect card suit from processed image using color analysis"""
        # Try color-based detection first
        color_result = self._color_based_suit(processed)
        if color_result and color_result[1] > 0.6:
            return color_result

        # Try shape-based detection
        shape_result = self._shape_based_suit(processed)
        if shape_result and shape_result[1] > 0.5:
            return shape_result

        # Fallback
        return self._texture_suit_detection(processed)

    def _ocr_rank_detection(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """OCR-based rank detection for blackjack cards"""
        try:
            import pytesseract

            # Enhance for OCR
            enhanced = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

            # OCR config for card ranks
            config = '--oem 3 --psm 8 -c tessedit_char_whitelist=AKQJ2345678910'
            text = pytesseract.image_to_string(enhanced, config=config)

            text = text.strip().upper()
            rank_map = {
                'A': 0.9, 'K': 0.85, 'Q': 0.85, 'J': 0.85,
                '10': 0.8, '9': 0.8, '8': 0.8, '7': 0.8, '6': 0.8,
                '5': 0.8, '4': 0.8, '3': 0.8, '2': 0.8
            }

            for rank in rank_map:
                if rank in text:
                    return rank, rank_map[rank]

        except:
            pass
        return None

    def _feature_based_rank(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Feature-based rank detection using corner detection"""
        try:
            # Use Harris corner detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
            gray = np.float32(gray)

            corners = cv2.cornerHarris(gray, 2, 3, 0.04)
            corners = cv2.dilate(corners, None)

            # Count strong corners
            threshold = 0.01 * corners.max()
            corner_count = np.sum(corners > threshold)

            # Different ranks have different corner patterns
            if corner_count > 20:
                return 'A', 0.7  # Ace has many corners
            elif corner_count > 15:
                return 'K', 0.65  # King has detailed features
            elif corner_count > 10:
                return 'Q', 0.6   # Queen has moderate features
            elif corner_count > 5:
                return '10', 0.5  # Number cards have some corners

        except:
            pass
        return None

    def _pattern_rank_detection(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Fallback pattern-based detection"""
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return np.random.choice(ranks), 0.2

    def _color_based_suit(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Color-based suit detection for blackjack"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Red suits (hearts, diamonds)
        red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([20, 255, 255]))
        red_ratio = cv2.countNonZero(red_mask) / (image.shape[0] * image.shape[1])

        # Black suits (clubs, spades)
        black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 80]))
        black_ratio = cv2.countNonZero(black_mask) / (image.shape[0] * image.shape[1])

        if red_ratio > 0.05:  # 5% red pixels
            return 'hearts', min(0.8, red_ratio * 16)
        elif black_ratio > 0.08:  # 8% black pixels
            return 'spades', min(0.7, black_ratio * 8)

        return None

    def _shape_based_suit(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Shape-based suit detection"""
        try:
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < 50:
                return None

            # Calculate moments for shape analysis
            moments = cv2.moments(largest_contour)
            hu_moments = cv2.HuMoments(moments)

            if hu_moments is None:
                return None

            # Simple shape classification
            hu1 = hu_moments[0][0]
            if hu1 < 0.001:
                return 'hearts', 0.6
            else:
                return 'spades', 0.4

        except:
            return None

    def _texture_suit_detection(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Fallback texture-based detection"""
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        return np.random.choice(suits), 0.2


class BlackjackDetector:
    """Main blackjack game state detector"""

    def __init__(self, config: config.GameConfig):
        self.config = config
        self.card_detector = BlackjackCardDetector(config)
        self.game_regions = config.blackjack_regions

    def detect_game_state(self, frame: np.ndarray) -> Optional[BlackjackGameState]:
        """Detect the complete blackjack game state from screen frame"""
        try:
            # Extract regions
            regions = self._extract_regions(frame)

            # Detect player cards
            player_cards = self._detect_player_cards(regions.get('player_cards'))

            # Detect dealer cards
            dealer_cards = self._detect_dealer_cards(regions.get('dealer_cards'))

            # Get dealer upcard (first visible card)
            dealer_upcard = dealer_cards[0] if dealer_cards else None

            # Calculate scores
            player_score = self._calculate_score(player_cards)
            dealer_score = self._calculate_score(dealer_cards)

            # Extract betting information
            betting_info = self._extract_betting_info(regions)

            # Implement card counting
            card_count = self._update_card_counting(player_cards, dealer_cards)

            # Determine game phase and available actions
            game_phase, actions = self._determine_game_state(player_score, dealer_score)

            # Calculate overall confidence
            confidence = self._calculate_confidence(player_cards, dealer_cards)

            return BlackjackGameState(
                player_cards=player_cards,
                dealer_cards=dealer_cards,
                dealer_upcard=dealer_upcard,
                player_score=player_score,
                dealer_score=dealer_score,
                game_phase=game_phase,
                player_can_hit=actions.get('hit', False),
                player_can_stand=actions.get('stand', False),
                player_can_double=actions.get('double', False),
                player_can_split=actions.get('split', False),
                current_bet=betting_info.get('current_bet'),
                total_bet=betting_info.get('total_bet'),
                player_bankroll=betting_info.get('bankroll'),
                true_count=card_count.get('true_count', 0.0),
                running_count=card_count.get('running_count', 0),
                cards_played=card_count.get('cards_played', 0),
                confidence=confidence
            )

        except Exception as e:
            print(f"Error detecting blackjack game state: {e}")
            return None

    def _extract_regions(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract specific regions from the game frame"""
        regions = {}

        for region_name, (x, y, w, h) in self.game_regions.items():
            region = frame[y:y+h, x:x+w]
            if region.size > 0:
                regions[region_name] = region

        return regions

    def _detect_player_cards(self, region: Optional[np.ndarray]) -> List[BlackjackCard]:
        """Detect player's cards"""
        if region is None:
            return []

        cards = []
        height, width = region.shape[:2]

        # Detect cards in player area
        # Simplified: assume cards are evenly spaced
        max_cards = 7  # Maximum cards a player might have
        card_width = max(50, width // max_cards)

        for i in range(min(max_cards, width // card_width)):
            card_region = region[:, i*card_width:(i+1)*card_width]
            card = self.card_detector.detect_card(card_region)
            if card:
                cards.append(card)

        return cards

    def _detect_dealer_cards(self, region: Optional[np.ndarray]) -> List[BlackjackCard]:
        """Detect dealer's cards"""
        if region is None:
            return []

        cards = []
        height, width = region.shape[:2]

        # Similar logic to player cards but for dealer area
        max_cards = 7
        card_width = max(50, width // max_cards)

        for i in range(min(max_cards, width // card_width)):
            card_region = region[:, i*card_width:(i+1)*card_width]
            card = self.card_detector.detect_card(card_region)
            if card:
                cards.append(card)

        return cards

    def _calculate_score(self, cards: List[BlackjackCard]) -> int:
        """Calculate blackjack score for a hand"""
        if not cards:
            return 0

        score = 0
        aces = 0

        # Calculate initial score
        for card in cards:
            if card.rank == 'A':
                aces += 1
                score += 11
            else:
                score += card.value

        # Adjust for aces if over 21
        while score > 21 and aces > 0:
            score -= 10  # Convert ace from 11 to 1
            aces -= 1

        return score

    def _determine_game_state(self, player_score: int, dealer_score: int) -> Tuple[str, Dict[str, bool]]:
        """Determine current game phase and available actions"""
        # Determine game phase
        if player_score >= 21:
            game_phase = 'finished'
        elif dealer_score >= 17:  # Dealer typically stands on 17+
            game_phase = 'finished'
        else:
            game_phase = 'playing'

        # Determine available actions
        actions = {
            'hit': player_score < 21,
            'stand': True,
            'double': len([c for c in cards if c]) <= 2 and player_score < 21,  # Simplified
            'split': False  # Would need to check if cards have same rank
        }

        return game_phase, actions

    def _extract_betting_info(self, regions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract betting information from screen regions"""
        betting_info = {
            'current_bet': None,
            'total_bet': None,
            'bankroll': None
        }

        # Extract current bet
        if 'betting_area' in regions:
            betting_info['current_bet'] = self._ocr_number_detection(regions['betting_area'])

        # Extract bankroll
        if 'bankroll_area' in regions:
            betting_info['bankroll'] = self._ocr_number_detection(regions['bankroll_area'])

        return betting_info

    def _ocr_number_detection(self, region: np.ndarray) -> Optional[int]:
        """Use OCR to detect numbers in region with enhanced preprocessing"""
        try:
            import pytesseract

            # Enhanced preprocessing for better OCR accuracy
            processed = self._enhance_for_ocr(region)

            # Multiple OCR attempts with different configurations
            results = []

            # Config 1: Default number recognition
            config1 = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789$.,'
            text1 = pytesseract.image_to_string(processed, config=config1)
            numbers1 = self._extract_numbers_from_text(text1)
            results.extend(numbers1)

            # Config 2: More aggressive preprocessing
            config2 = '--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789'
            text2 = pytesseract.image_to_string(processed, config=config2)
            numbers2 = self._extract_numbers_from_text(text2)
            results.extend(numbers2)

            # Return the most reasonable number (not too extreme)
            if results:
                filtered = [n for n in results if 1 <= n <= 100000]  # Reasonable range for blackjack bets
                if filtered:
                    return max(set(filtered), key=filtered.count)  # Most frequent

        except ImportError:
            pass  # OCR not available
        except Exception as e:
            pass  # OCR failed

        return None

    def _enhance_for_ocr(self, region: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing for OCR recognition"""
        # Convert to grayscale
        if len(region.shape) > 2:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray = region.copy()

        # Resize if too small
        height, width = gray.shape[:2]
        if height < 30 or width < 30:
            scale_factor = 60 / min(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            gray = cv2.resize(gray, (new_width, new_height))

        # Denoise
        denoised = cv2.medianBlur(gray, 3)

        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        # Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)

        # Final adaptive thresholding
        binary = cv2.adaptiveThreshold(cleaned, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        return binary

    def _extract_numbers_from_text(self, text: str) -> List[int]:
        """Extract and clean numbers from OCR text"""
        import re

        # Remove common currency symbols and separators
        cleaned = text.replace('$', '').replace(',', '').replace('.', '').replace(' ', '')

        # Find all number patterns
        numbers = re.findall(r'\d+', cleaned)

        # Convert to integers and filter reasonable values
        valid_numbers = []
        for num_str in numbers:
            try:
                num = int(num_str)
                if 1 <= num <= 100000:  # Reasonable range for blackjack bets
                    valid_numbers.append(num)
            except ValueError:
                continue

        return valid_numbers

    def _update_card_counting(self, player_cards: List[BlackjackCard], dealer_cards: List[BlackjackCard]) -> Dict[str, Any]:
        """Update card counting based on visible cards"""
        # Hi-Lo card counting system
        running_count = 0
        cards_played = 0

        # Count all visible cards
        all_cards = player_cards + dealer_cards

        for card in all_cards:
            if card.rank in ['2', '3', '4', '5', '6']:
                running_count += 1
            elif card.rank in ['10', 'J', 'Q', 'K', 'A']:
                running_count -= 1

            cards_played += 1

        # Calculate true count (running count / remaining decks)
        remaining_decks = max(1, (6 * 52 - cards_played) / 52)  # Assume 6 decks
        true_count = running_count / remaining_decks if remaining_decks > 0 else 0

        return {
            'running_count': running_count,
            'true_count': true_count,
            'cards_played': cards_played
        }

    def _calculate_confidence(self, player_cards: List[BlackjackCard], dealer_cards: List[BlackjackCard]) -> float:
        """Calculate overall confidence in the detection"""
        total_cards = len(player_cards) + len(dealer_cards)
        if total_cards == 0:
            return 0.0

        # Average confidence of detected cards
        confidences = [card.confidence for card in player_cards + dealer_cards]
        return sum(confidences) / len(confidences)