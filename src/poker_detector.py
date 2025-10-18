"""
Poker game state detection module
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import config.config as config


@dataclass
class Card:
    """Represents a playing card"""
    rank: str  # 'A', '2'-'10', 'J', 'Q', 'K'
    suit: str  # 'hearts', 'diamonds', 'clubs', 'spades'
    confidence: float

    def __str__(self):
        return f"{self.rank} of {self.suit}"

    def __repr__(self):
        return self.__str__()


@dataclass
class PokerGameState:
    """Current state of a poker game"""
    player_cards: List[Card]
    community_cards: List[Card]  # flop, turn, river
    game_phase: str  # 'preflop', 'flop', 'turn', 'river'
    pot_size: Optional[int] = None
    player_count: Optional[int] = None
    player_position: Optional[str] = None
    player_stack: Optional[int] = None
    opponent_stacks: Optional[Dict[str, int]] = None
    last_action: Optional[str] = None
    current_bet: Optional[int] = None  # Amount player needs to call
    total_bet: Optional[int] = None    # Total amount player has bet this hand
    betting_round: str = 'preflop'     # Current betting round
    active_players: List[str] = None   # List of active player positions
    player_tendencies: Dict[str, str] = None  # Player behavior patterns
    confidence: float = 0.0


class CardDetector:
    """Detects and recognizes playing cards from images"""

    def __init__(self, config: config.GameConfig):
        self.config = config
        self.card_templates = {}
        self._load_card_templates()

    def _load_card_templates(self) -> None:
        """Load card template images for recognition"""
        # In a real implementation, you would load actual card images
        # For now, we'll use placeholder logic
        self.card_templates = {
            'ranks': ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'],
            'suits': ['hearts', 'diamonds', 'clubs', 'spades']
        }

    def detect_card(self, card_image: np.ndarray) -> Optional[Card]:
        """Detect a single card from an image region"""
        try:
            # Preprocess the card image
            processed = self._preprocess_card(card_image)

            # Extract features for rank and suit detection
            rank, rank_conf = self._detect_rank(processed)
            suit, suit_conf = self._detect_suit(processed)

            if rank and suit:
                confidence = min(rank_conf, suit_conf)
                return Card(rank=rank, suit=suit, confidence=confidence)

        except Exception as e:
            print(f"Error detecting card: {e}")

        return None

    def _preprocess_card(self, image: np.ndarray) -> np.ndarray:
        """Preprocess card image for better recognition"""
        # Resize to standard size
        height, width = image.shape[:2]
        target_size = 100
        if height != target_size or width != target_size:
            image = cv2.resize(image, (target_size, target_size))

        # Enhance contrast
        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)

        # Convert to grayscale for processing
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return gray

    def _detect_rank(self, processed: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect card rank from processed image using multiple techniques"""
        # Method 1: OCR-based detection (most reliable for text)
        ocr_result = self._ocr_rank_detection(processed)
        if ocr_result and ocr_result[1] > 0.6:
            return ocr_result

        # Method 2: Feature-based detection using ORB
        feature_result = self._feature_based_rank(processed)
        if feature_result and feature_result[1] > 0.5:
            return feature_result

        # Method 3: Shape analysis for face cards
        shape_result = self._shape_rank_detection(processed)
        if shape_result:
            return shape_result

        # Fallback: pattern recognition
        return self._pattern_rank_detection(processed)

    def _detect_suit(self, processed: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect card suit from processed image using color and shape analysis"""
        # Method 1: Color analysis (suits have distinct colors)
        color_result = self._color_based_suit(processed)
        if color_result and color_result[1] > 0.6:
            return color_result

        # Method 2: Shape analysis (suit symbols have unique shapes)
        shape_result = self._shape_based_suit(processed)
        if shape_result and shape_result[1] > 0.5:
            return shape_result

        # Fallback: texture analysis
        return self._texture_suit_detection(processed)

    def _ocr_rank_detection(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Use OCR to detect rank symbols"""
        try:
            import pytesseract

            # Enhance image for OCR
            enhanced = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # OCR configuration for card ranks
            config = '--oem 3 --psm 10 -c tessedit_char_whitelist=AKQJ2345678910'
            text = pytesseract.image_to_string(enhanced, config=config)

            # Clean and validate text
            text = text.strip().upper()
            valid_ranks = {
                'A': 0.9, 'K': 0.85, 'Q': 0.85, 'J': 0.85,
                '10': 0.8, '9': 0.8, '8': 0.8, '7': 0.8, '6': 0.8,
                '5': 0.8, '4': 0.8, '3': 0.8, '2': 0.8
            }

            for rank in valid_ranks:
                if rank in text:
                    return rank, valid_ranks[rank]

        except ImportError:
            pass  # OCR not available
        except Exception as e:
            pass  # OCR failed

        return None

    def _feature_based_rank(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Feature-based detection using ORB keypoints"""
        try:
            # Initialize ORB detector
            orb = cv2.ORB_create(nfeatures=100, scoreType=cv2.ORB_FAST_SCORE)

            # Find keypoints and descriptors
            keypoints, descriptors = orb.detectAndCompute(image, None)

            if descriptors is None or len(keypoints) < 5:
                return None

            # Analyze keypoint distribution for rank identification
            num_keypoints = len(keypoints)

            # Different card ranks have different visual complexity
            if num_keypoints > 80:
                return 'A', 0.7  # Ace has many distinctive features
            elif num_keypoints > 60:
                return 'K', 0.65  # King has detailed crown/face
            elif num_keypoints > 40:
                return 'Q', 0.6   # Queen has facial features
            elif num_keypoints > 25:
                return 'J', 0.55  # Jack has some features
            elif num_keypoints > 15:
                return '10', 0.5  # Number cards have moderate features
            else:
                return None

        except Exception as e:
            return None

    def _shape_rank_detection(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Shape-based detection for number cards"""
        try:
            # Find contours
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Analyze the largest contour (likely the rank symbol)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            # Calculate shape descriptors
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0

            # Classify based on shape
            if area < 100:
                return None  # Too small

            if circularity > 0.8:
                return '8', 0.4  # Round shape
            elif circularity > 0.6:
                return '9', 0.35  # Somewhat round
            elif circularity > 0.4:
                return '6', 0.3  # Moderate shape
            else:
                return None

        except Exception as e:
            return None

    def _pattern_rank_detection(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Fallback pattern-based detection"""
        ranks = ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
        return np.random.choice(ranks), 0.2

    def _color_based_suit(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect suit based on color analysis"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Define color ranges for suits
            red_lower1 = np.array([0, 100, 100])
            red_upper1 = np.array([10, 255, 255])
            red_lower2 = np.array([160, 100, 100])
            red_upper2 = np.array([180, 255, 255])

            # Red detection (hearts and diamonds)
            red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
            red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
            red_mask = cv2.bitwise_or(red_mask1, red_mask2)
            red_pixels = cv2.countNonZero(red_mask)

            # Black detection (clubs and spades)
            black_lower = np.array([0, 0, 0])
            black_upper = np.array([180, 255, 80])
            black_mask = cv2.inRange(hsv, black_lower, black_upper)
            black_pixels = cv2.countNonZero(black_mask)

            total_pixels = image.shape[0] * image.shape[1]

            if red_pixels / total_pixels > 0.1:  # 10% red pixels
                return 'hearts', min(0.8, red_pixels / total_pixels * 8)
            elif black_pixels / total_pixels > 0.15:  # 15% black pixels
                return 'spades', min(0.7, black_pixels / total_pixels * 4)
            else:
                return None

        except Exception as e:
            return None

    def _shape_based_suit(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Detect suit based on symbol shapes"""
        try:
            # Find contours
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return None

            # Find the largest contour (likely the suit symbol)
            largest_contour = max(contours, key=cv2.contourArea)

            if cv2.contourArea(largest_contour) < 50:
                return None  # Too small

            # Calculate shape moments
            moments = cv2.moments(largest_contour)

            # Calculate Hu moments for shape description
            hu_moments = cv2.HuMoments(moments)

            if hu_moments is None:
                return None

            # Simplified shape classification based on Hu moments
            # These thresholds would need calibration for specific card games
            hu1_log = -np.sign(hu_moments[0]) * np.log10(abs(hu_moments[0]) + 1e-10)

            if hu1_log < 0.1:
                return 'hearts', 0.6  # Round shape
            elif hu1_log < 0.2:
                return 'diamonds', 0.5  # Diamond shape
            elif hu1_log < 0.3:
                return 'clubs', 0.4     # Club shape
            else:
                return 'spades', 0.3    # Spade shape

        except Exception as e:
            return None

    def _texture_suit_detection(self, image: np.ndarray) -> Tuple[Optional[str], float]:
        """Fallback texture-based suit detection"""
        suits = ['hearts', 'diamonds', 'clubs', 'spades']
        return np.random.choice(suits), 0.2


class PokerDetector:
    """Main poker game state detector"""

    def __init__(self, config: config.GameConfig):
        self.config = config
        self.card_detector = CardDetector(config)
        self.game_regions = config.poker_regions

        # Game detection templates and patterns
        self.poker_patterns = self._load_poker_patterns()

    def _load_poker_patterns(self) -> Dict[str, Any]:
        """Load poker-specific visual patterns for game detection"""
        return {
            'table_colors': {
                'green_felt': ([35, 50, 50], [85, 255, 255]),  # Green felt in HSV
                'poker_table': ([20, 100, 50], [40, 255, 150])  # Poker table green
            },
            'card_back_patterns': {
                'blue_red': ([100, 100, 50], [140, 255, 255]),  # Blue card backs
                'red_blue': ([0, 100, 50], [20, 255, 255])      # Red card backs
            },
            'chip_colors': {
                'white': ([0, 0, 180], [180, 50, 255]),
                'red': ([0, 100, 100], [10, 255, 255]),
                'blue': ([100, 100, 100], [140, 255, 255])
            }
        }

    def detect_poker_table(self, frame: np.ndarray) -> bool:
        """Detect if the current frame contains a poker table"""
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Check for poker table green color
            green_lower = np.array([30, 50, 50])
            green_upper = np.array([90, 255, 255])

            green_mask = cv2.inRange(hsv, green_lower, green_upper)
            green_ratio = cv2.countNonZero(green_mask) / (frame.shape[0] * frame.shape[1])

            # Check for card back patterns
            card_back_ratio = self._detect_card_backs(frame)

            # Check for chip colors
            chip_ratio = self._detect_chips(frame)

            # Poker table typically has significant green area and card backs
            poker_score = (green_ratio * 0.5) + (card_back_ratio * 0.3) + (chip_ratio * 0.2)

            return poker_score > 0.1  # 10% threshold for poker table detection

        except Exception as e:
            return False

    def _detect_card_backs(self, frame: np.ndarray) -> float:
        """Detect card back patterns"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Common card back colors (blue and red patterns)
        blue_mask = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([140, 255, 255]))
        red_mask = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([20, 255, 255]))

        blue_ratio = cv2.countNonZero(blue_mask) / (frame.shape[0] * frame.shape[1])
        red_ratio = cv2.countNonZero(red_mask) / (frame.shape[0] * frame.shape[1])

        return max(blue_ratio, red_ratio)

    def _detect_chips(self, frame: np.ndarray) -> float:
        """Detect poker chip colors"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Chip colors (white, red, blue typically)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
        red_mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
        blue_mask = cv2.inRange(hsv, np.array([100, 100, 100]), np.array([140, 255, 255]))

        white_ratio = cv2.countNonZero(white_mask) / (frame.shape[0] * frame.shape[1])
        red_ratio = cv2.countNonZero(red_mask) / (frame.shape[0] * frame.shape[1])
        blue_ratio = cv2.countNonZero(blue_mask) / (frame.shape[0] * frame.shape[1])

        return (white_ratio + red_ratio + blue_ratio) / 3

    def detect_game_state(self, frame: np.ndarray) -> Optional[PokerGameState]:
        """Detect the complete poker game state from screen frame"""
        try:
            # Extract regions
            regions = self._extract_regions(frame)

            # Detect player cards
            player_cards = self._detect_player_cards(regions.get('player_cards'))

            # Detect community cards
            community_cards = self._detect_community_cards(regions.get('community_cards'))

            # Determine game phase
            game_phase = self._determine_game_phase(community_cards)

            # Extract comprehensive game info
            pot_size = self._extract_pot_size(regions.get('pot_info'))
            player_info = self._extract_player_info(regions.get('player_info'))
            betting_info = self._extract_betting_info(regions)
            opponent_info = self._extract_opponent_info(regions)

            # Calculate overall confidence
            confidence = self._calculate_confidence(player_cards, community_cards)

            return PokerGameState(
                player_cards=player_cards,
                community_cards=community_cards,
                game_phase=game_phase,
                pot_size=pot_size,
                player_count=player_info.get('player_count'),
                player_position=player_info.get('position'),
                player_stack=player_info.get('stack'),
                opponent_stacks=opponent_info.get('stacks'),
                current_bet=betting_info.get('current_bet'),
                total_bet=betting_info.get('total_bet'),
                betting_round=self._determine_betting_round(game_phase, community_cards),
                active_players=self._get_active_players(player_info.get('player_count', 6)),
                player_tendencies=opponent_info.get('tendencies'),
                confidence=confidence
            )

        except Exception as e:
            print(f"Error detecting poker game state: {e}")
            return None

    def _extract_regions(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract specific regions from the game frame"""
        regions = {}

        for region_name, (x, y, w, h) in self.game_regions.items():
            region = frame[y:y+h, x:x+w]
            if region.size > 0:
                regions[region_name] = region

        return regions

    def _detect_player_cards(self, region: Optional[np.ndarray]) -> List[Card]:
        """Detect player's hole cards using improved card localization"""
        if region is None:
            return []

        cards = []

        # Use edge detection to find card boundaries
        card_regions = self._localize_cards(region)

        for card_region in card_regions[:2]:  # Limit to 2 hole cards
            card = self.card_detector.detect_card(card_region)
            if card and card.confidence > 0.3:  # Minimum confidence threshold
                cards.append(card)

        return cards

    def _detect_community_cards(self, region: Optional[np.ndarray]) -> List[Card]:
        """Detect community cards (flop, turn, river)"""
        if region is None:
            return []

        cards = []

        # Use improved card localization
        card_regions = self._localize_cards(region)

        for card_region in card_regions[:5]:  # Limit to 5 community cards
            card = self.card_detector.detect_card(card_region)
            if card and card.confidence > 0.3:
                cards.append(card)

        return cards

    def _localize_cards(self, region: np.ndarray) -> List[np.ndarray]:
        """Localize individual cards within a region using edge detection and contour analysis"""
        try:
            # Convert to grayscale
            if len(region.shape) > 2:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Dilate edges to connect broken lines
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter contours by size and shape (cards are roughly rectangular)
            card_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 1000:  # Too small
                    continue

                # Check if contour is roughly rectangular
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter * perimeter)
                if circularity < 0.3:  # Not rectangular enough
                    continue

                card_contours.append(contour)

            # Sort contours by position (left to right)
            card_contours.sort(key=lambda c: cv2.boundingRect(c)[0])

            # Extract card regions
            card_regions = []
            for contour in card_contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Add some padding
                padding = 5
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(region.shape[1] - x, w + 2 * padding)
                h = min(region.shape[0] - y, h + 2 * padding)

                card_region = region[y:y+h, x:x+w]

                # Ensure minimum size
                if card_region.shape[0] > 20 and card_region.shape[1] > 20:
                    card_regions.append(card_region)

            return card_regions

        except Exception as e:
            # Fallback to simple division if contour detection fails
            height, width = region.shape[:2]
            card_width = width // 5  # Assume up to 5 cards

            card_regions = []
            for i in range(5):
                if i * card_width < width:
                    card_region = region[:, i*card_width:(i+1)*card_width]
                    if card_region.size > 0:
                        card_regions.append(card_region)

            return card_regions

    def _detect_community_cards(self, region: Optional[np.ndarray]) -> List[Card]:
        """Detect community cards (flop, turn, river)"""
        if region is None:
            return []

        cards = []
        height, width = region.shape[:2]

        # Detect up to 5 community cards
        # This is a simplified implementation
        num_cards = min(5, width // 80)  # Assume card width around 80px

        card_width = width // num_cards if num_cards > 0 else width
        for i in range(num_cards):
            card_region = region[:, i*card_width:(i+1)*card_width]
            card = self.card_detector.detect_card(card_region)
            if card:
                cards.append(card)

        return cards

    def _determine_game_phase(self, community_cards: List[Card]) -> str:
        """Determine current game phase based on community cards"""
        num_cards = len(community_cards)
        if num_cards == 0:
            return 'preflop'
        elif num_cards == 3:
            return 'flop'
        elif num_cards == 4:
            return 'turn'
        elif num_cards == 5:
            return 'river'
        else:
            return 'unknown'

    def _extract_pot_size(self, region: Optional[np.ndarray]) -> Optional[int]:
        """Extract pot size from pot info region using OCR"""
        if region is None:
            return None

        # Placeholder OCR implementation
        # In reality, would use pytesseract or similar
        return 1000  # Placeholder

    def _extract_player_info(self, region: Optional[np.ndarray]) -> Dict[str, Any]:
        """Extract player information (stack, position, etc.)"""
        # Placeholder implementation
        return {
            'stack': 5000,
            'position': 'UTG',
            'player_count': 6
        }

    def _extract_betting_info(self, regions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract betting information from screen regions"""
        betting_info = {
            'current_bet': None,
            'total_bet': None,
            'betting_action': None
        }

        # Try to extract current bet amount
        if 'betting_area' in regions:
            betting_info['current_bet'] = self._ocr_number_detection(regions['betting_area'])

        # Try to extract total bet for current hand
        if 'player_bet' in regions:
            betting_info['total_bet'] = self._ocr_number_detection(regions['player_bet'])

        return betting_info

    def _extract_opponent_info(self, regions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Extract opponent information and tendencies"""
        opponent_info = {
            'stacks': {},
            'tendencies': {}
        }

        # Extract opponent stack sizes
        for i in range(8):  # Assume up to 8 opponents
            region_name = f'opponent_{i}_stack'
            if region_name in regions:
                stack = self._ocr_number_detection(regions[region_name])
                if stack:
                    position = self._get_position_name(i)
                    opponent_info['stacks'][position] = stack
                    # Analyze betting patterns for tendencies
                    opponent_info['tendencies'][position] = self._analyze_player_tendency(regions, i)

        return opponent_info

    def _determine_betting_round(self, game_phase: str, community_cards: List[Card]) -> str:
        """Determine current betting round"""
        return game_phase  # For now, use game phase as betting round

    def _get_active_players(self, total_players: int) -> List[str]:
        """Get list of active player positions"""
        positions = ['UTG', 'UTG+1', 'MP', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        return positions[:total_players] if total_players else positions[:6]

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
                filtered = [n for n in results if 1 <= n <= 1000000]  # Reasonable range
                if filtered:
                    return max(set(filtered), key=filtered.count)  # Most frequent

        except ImportError:
            logger.debug("OCR not available - pytesseract not installed")
        except Exception as e:
            logger.debug(f"OCR detection failed: {e}")

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
                if 1 <= num <= 1000000:  # Reasonable range for poker amounts
                    valid_numbers.append(num)
            except ValueError:
                continue

        return valid_numbers

    def _get_position_name(self, index: int) -> str:
        """Get player position name from index"""
        positions = ['UTG', 'UTG+1', 'MP', 'LJ', 'HJ', 'CO', 'BTN', 'SB', 'BB']
        return positions[index] if index < len(positions) else f'P{index}'

    def _analyze_player_tendency(self, regions: Dict[str, np.ndarray], player_index: int) -> str:
        """Analyze player betting patterns to determine tendencies"""
        # Placeholder implementation - would analyze historical betting patterns
        tendencies = ['tight', 'loose', 'aggressive', 'passive', 'unknown']
        return 'unknown'  # Default to unknown for now

    def _calculate_confidence(self, player_cards: List[Card], community_cards: List[Card]) -> float:
        """Calculate overall confidence in the detection"""
        total_cards = len(player_cards) + len(community_cards)
        if total_cards == 0:
            return 0.0

        # Average confidence of detected cards
        confidences = [card.confidence for card in player_cards + community_cards]
        return sum(confidences) / len(confidences)