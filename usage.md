# 📖 Usage Guide

## Practical Usage Examples and Scenarios

This guide provides practical examples of how to use the AI Blackjack Poker Assistant in real gaming scenarios, best practices, and advanced techniques.

## Table of Contents

1. [Poker Usage Examples](#poker-usage-examples)
2. [Blackjack Usage Examples](#blackjack-usage-examples)
3. [Multi-Table Scenarios](#multi-table-scenarios)
4. [Tournament Play](#tournament-play)
5. [Bankroll Management](#bankroll-management)
6. [Session Analysis](#session-analysis)
7. [Advanced Techniques](#advanced-techniques)
8. [Best Practices](#best-practices)

## Poker Usage Examples

### Scenario 1: Cash Game - Early Position with Premium Hand

**Game State:**
- Position: UTG (Under the Gun)
- Hole Cards: A♠ A♦ (Pocket Aces)
- Stack: $200, Blinds: $1/$2
- Table: 6 players, mostly recreational

**AI Analysis Display:**
```
┌─────────────────────────────────────┐
│ POKER ANALYSIS                      │
│ Hand: A♠ A♦ (Pocket Aces)          │
│ Strength: Monster (95% vs random)   │
│ Equity vs Range: 82%               │
│ Pot Odds: N/A (preflop)            │
│ Recommended: RAISE (4x)            │
│ Position: UTG                      │
└─────────────────────────────────────┘
```

**Explanation:**
- Pocket Aces are the strongest starting hand (Monster category)
- 95% equity means you'll win most hands against random holdings
- In early position, raise to 4x big blind ($8) to build the pot
- Be prepared for multiple callers given position

**Action Plan:**
1. Open raise to $8-$10
2. If 3-bet, consider 4-betting for value
3. Post-flop: Play straightforward, value bet strong boards

### Scenario 2: Cash Game - Late Position with Drawing Hand

**Game State:**
- Position: Button
- Hole Cards: 8♥ 7♥ (Suited Connectors)
- Flop: A♦ 6♣ 5♥ (rainbow)
- Pot: $15, 3 players to act

**AI Analysis Display:**
```
┌─────────────────────────────────────┐
│ POKER ANALYSIS                      │
│ Hand: 8♥ 7♥ (Gutshot + Backdoor)   │
│ Strength: Medium (45% vs random)   │
│ Outs: 4 (9,8,7,4 for straight)     │
│ Pot Odds: 2:1 (33% required)       │
│ Recommended: CALL                  │
│ Position: BTN (Strong)             │
└─────────────────────────────────────┘
```

**Explanation:**
- 8♥7♥ has gutshot straight draw (needs 9 for straight)
- Backdoor flush potential (two hearts)
- 45% equity makes calling profitable at 2:1 odds
- Button position provides excellent bluffing opportunity

**Action Plan:**
1. Call the flop bet
2. On turn: Evaluate based on new cards and action
3. Use position to control pot size
4. Bluff catch if villain shows weakness

### Scenario 3: Tournament - Bubble Play with Short Stack

**Game State:**
- Tournament: 180 players, 20 remaining (bubble)
- Position: Middle
- Stack: 8 BB, Blinds: 1000/2000
- Hole Cards: K♠ Q♦

**AI Analysis Display:**
```
┌─────────────────────────────────────┐
│ POKER ANALYSIS                      │
│ Hand: K♠ Q♦ (Broadway Cards)      │
│ Strength: Strong (62% vs random)   │
│ Stack: 8 BB (Critical)             │
│ Pot Odds: All-in situation         │
│ Recommended: ALL-IN                │
│ Position: MP                       │
└─────────────────────────────────────┘
```

**Explanation:**
- KQ is strong enough to shove 8BB profitably
- 62% equity vs random range makes this +EV
- Bubble pressure increases fold equity
- Can't afford to wait for premium hands

**Action Plan:**
1. Open shove remaining stack
2. Target players with 15-40 BB stacks
3. Avoid table chip leader if possible
4. Prepare to call if reshoved by similar stack

## Blackjack Usage Examples

### Scenario 1: Single Deck - Basic Strategy with Count

**Game State:**
- Casino: Single deck, H17, DAS allowed
- Player Hand: 16 (10 + 6) vs Dealer 7
- True Count: +3 (favorable)
- Bankroll: $500, Bet: $10

**AI Analysis Display:**
```
┌─────────────────────────────────────┐
│ BLACKJACK ANALYSIS                  │
│ Hand: 16 vs Dealer 7               │
│ Win Probability: 38%               │
│ Dealer Bust: 42%                   │
│ True Count: +3 (Favorable)         │
│ Recommended: STAND (Deviation)     │
│ Running Count: +12                 │
└─────────────────────────────────────┘
```

**Explanation:**
- Basic strategy says HIT 16 vs 7
- True count +3 favors standing (research-based deviation)
- Favorable count means dealer more likely to bust
- Expected value of standing: +$1.20 vs hitting: -$0.80

**Action Plan:**
1. Stand on 16 (count-based deviation)
2. Increase bet size next hand due to favorable count
3. Track penetration for bet sizing
4. Consider insurance if dealer shows Ace

### Scenario 2: Multi-Deck - Soft Hand Decision

**Game State:**
- Casino: 6 decks, S17, DAS allowed
- Player Hand: A-7 (Soft 18) vs Dealer 9
- True Count: -1 (unfavorable)
- Current Bet: $25

**AI Analysis Display:**
```
┌─────────────────────────────────────┐
│ BLACKJACK ANALYSIS                  │
│ Hand: A-7 (Soft 18) vs Dealer 9    │
│ Win Probability: 45%               │
│ Dealer Bust: 23%                   │
│ True Count: -1 (Unfavorable)       │
│ Recommended: STAND                 │
│ Strategy: Basic Strategy           │
└─────────────────────────────────────┘
```

**Explanation:**
- Soft 18 vs 9: Basic strategy is STAND
- Negative count reduces advantage
- Doubling would be too risky vs strong dealer card
- Standing gives best chance to win

**Action Plan:**
1. Stand on soft 18
2. Reduce bet size due to unfavorable count
3. Look for surrender opportunities if available
4. Track for more favorable situations

### Scenario 3: Card Counting - Favorable Shoe

**Game State:**
- Casino: 6 decks, penetration at 75%
- True Count: +5 (very favorable)
- Player Hand: 10-10 vs Dealer 6
- Current Bet: $10

**AI Analysis Display:**
```
┌─────────────────────────────────────┐
│ BLACKJACK ANALYSIS                  │
│ Hand: 20 vs Dealer 6               │
│ Win Probability: 78%               │
│ True Count: +5 (Very Favorable)    │
│ Recommended: STAND                 │
│ Bet Size: MAXIMUM ($100)           │
└─────────────────────────────────────┘
```

**Explanation:**
- 20 is automatic stand regardless of count
- True count +5 indicates deck rich in high cards
- Optimal bet sizing: 10x base bet ($100)
- Kelly Criterion suggests maximum advantage

**Action Plan:**
1. Stand on 20 (obvious)
2. Increase bet to $100 next hand
3. Continue aggressive betting until count drops
4. Monitor for casino heat/backoffs

## Multi-Table Scenarios

### Scenario 1: Two Poker Tables

**Setup:**
- Table 1: $1/$2 NLHE, 6 players
- Table 2: $0.50/$1 NLHE, 9 players
- Total Bankroll: $800

**AI Overlay Management:**
```
┌─────────────────────────────────────┐
│ MULTI-TABLE ANALYSIS               │
│ Table 1: $1/$2 - Hand in Progress  │
│ Table 2: $0.50/$1 - Waiting       │
│ Active: Table 1                    │
│ Combined Profit: +$45             │
└─────────────────────────────────────┘
```

**Management Strategy:**
1. Focus on higher stakes table when both active
2. Use table with better position when available
3. Monitor overall session profit across tables
4. Take breaks between table changes

### Scenario 2: Mixed Poker and Blackjack

**Setup:**
- Poker: $0.50/$1 NLHE
- Blackjack: $10 minimum bet

**AI Display:**
```
┌─────────────────────────────────────┐
│ MIXED GAME ANALYSIS                │
│ Active: Blackjack                  │
│ Poker Tables: 2 (Waiting)          │
│ BJ Count: +2 (Favorable)           │
│ Session EV: +$67                   │
└─────────────────────────────────────┘
```

**Strategy:**
1. Play blackjack during favorable counts
2. Switch to poker when count is neutral/negative
3. Monitor overall bankroll across games
4. Use blackjack winnings to fund poker play

## Tournament Play

### Scenario 1: Early Tournament - Deep Stack

**Tournament State:**
- Players: 180/200 remaining
- Stack: 50 BB
- Position: Middle
- Hand: J♥ 10♥ (Suited Broadway)

**AI Analysis:**
```
┌─────────────────────────────────────┐
│ TOURNAMENT ANALYSIS                │
│ Hand: J♥ 10♥ (Suited Connectors)   │
│ Stack: 50 BB (Deep)                │
│ Position: MP                       │
│ Tournament: Early Stage            │
│ Recommended: CALL/RAISE            │
└─────────────────────────────────────┘
```

**Strategy:**
- Deep stacks allow for post-flop play
- Suited connectors play well multi-way
- Position provides fold equity for bluffs
- Accumulate chips steadily in early stages

### Scenario 2: Late Tournament - ICM Pressure

**Tournament State:**
- Players: 12/180 remaining (final table bubble)
- Stack: 15 BB
- Position: Button
- Hand: A♦ Q♠

**AI Analysis:**
```
┌─────────────────────────────────────┐
│ TOURNAMENT ANALYSIS                │
│ Hand: A♦ Q♠ (Premium)             │
│ Stack: 15 BB (Medium)              │
│ Position: BTN (Excellent)          │
│ ICM: High Pressure                 │
│ Recommended: RAISE (3x)            │
└─────────────────────────────────────┘
```

**Strategy:**
- AQ is premium hand worth playing aggressively
- Button position provides excellent fold equity
- ICM considerations make tight folds correct
- Pressure short stacks when possible

## Bankroll Management

### Real-Time Bankroll Tracking

**Session Overview:**
```
┌─────────────────────────────────────┐
│ BANKROLL MANAGEMENT                │
│ Starting Bankroll: $1000           │
│ Current: $1245 (+$245, +24.5%)    │
│ Session Length: 2h 15m             │
│ Hourly Rate: +$109                 │
│ Risk of Ruin: 2.1%                 │
└─────────────────────────────────────┘
```

**Management Rules:**
1. **Stop-Loss:** Quit if down 20% of starting bankroll
2. **Stop-Win:** Quit if up 50% of starting bankroll
3. **Time Limits:** Play maximum 3 hours per session
4. **Game Selection:** Only play +EV games

### Bet Sizing Optimization

**Blackjack Bet Sizing:**
- True Count +1: 2x minimum bet
- True Count +3: 5x minimum bet
- True Count +5: 10x minimum bet
- Maximum bet: 5% of bankroll

**Poker Bet Sizing:**
- Value bets: 50-75% pot
- Bluff bets: 33-50% pot
- Tournament shoves: Based on ICM and fold equity

## Session Analysis

### Post-Session Review

**Performance Metrics:**
```
┌─────────────────────────────────────┐
│ SESSION ANALYSIS                   │
│ Game: Blackjack                    │
│ Duration: 3 hours                  │
│ Hands Played: 180                  │
│ Profit/Loss: +$340                 │
│ Hourly Rate: +$113                 │
│ True Count Average: +1.2           │
└─────────────────────────────────────┘
```

**Analysis Points:**
1. **Strategy Effectiveness:** Review hands where AI recommendations differed from play
2. **Count Accuracy:** Compare estimated true count vs actual
3. **Bet Sizing:** Evaluate if bet sizes were optimal
4. **Game Selection:** Assess if table/casino choice was good

### Historical Trend Analysis

**Monthly Performance:**
```
┌─────────────────────────────────────┐
│ MONTHLY PERFORMANCE                │
│ Total Sessions: 12                 │
│ Win Rate: 75% (9/12 profitable)    │
│ Average Session: +$156             │
│ Best Session: +$420                │
│ Worst Session: -$85                │
│ Standard Deviation: $145           │
└─────────────────────────────────────┘
```

**Trend Analysis:**
- Improving month over month
- Best performance in blackjack
- Poker results more volatile
- Consistent improvement in decision-making

## Advanced Techniques

### Custom Strategy Adjustments

#### Personalized Playing Style

```python
# Aggressive style configuration
aggressive_config = {
    'poker_style': {
        'vpip_target': 0.24,           # Higher VPIP
        'pfr_target': 0.20,            # Higher PFR
        'aggression_factor': 2.5       # More aggressive
    },
    'blackjack_style': {
        'risk_tolerance': 'high',      # Take more chances
        'bet_sizing': 'kelly_plus',    # Larger bets
        'deviations': 'aggressive'     # More deviations
    }
}
```

#### Situation-Specific Adjustments

**Tournament Bubble Play:**
- Increase fold equity calculations
- Adjust ranges for ICM pressure
- Consider ladder implications
- Modify bet sizing for bubble

**Short-Handed Play:**
- Widen ranges significantly
- Increase bluff frequency
- Adjust for higher variance
- Modify pot odds calculations

### Live Tracking and Adjustments

#### Real-Time Performance Monitoring

**During Session:**
1. Monitor AI accuracy vs actual results
2. Track bankroll movement
3. Note count accuracy
4. Adjust strategy based on observations

**Adjustment Triggers:**
- If AI is consistently wrong: Re-evaluate ranges
- If count seems off: Re-check deck estimation
- If losing consistently: Review game selection
- If fatigued: Take mandatory breaks

## Best Practices

### Pre-Session Preparation

1. **Bankroll Check**
   - Ensure adequate bankroll for stakes
   - Set stop-loss and stop-win limits
   - Have emergency fund separate

2. **Technical Setup**
   - Test calibration before playing
   - Verify overlay positioning
   - Check internet connection stability

3. **Game Selection**
   - Choose tables with recreational players
   - Look for favorable blackjack rules/penetration
   - Avoid tables with known professionals

### During Session

1. **Focus and Attention**
   - Avoid distractions and multi-tasking
   - Take regular breaks (5 minutes/hour)
   - Stay hydrated and comfortable

2. **Decision Making**
   - Trust AI recommendations but verify logic
   - Consider table dynamics and opponent tendencies
   - Don't override AI without good reason

3. **Bankroll Management**
   - Never risk more than 5% per hand
   - Adjust bet sizes based on count/confidence
   - Quit when stop-loss reached

### Post-Session

1. **Immediate Review**
   - Note major hands and decisions
   - Record profit/loss and time played
   - Identify mistakes and learning points

2. **Detailed Analysis**
   - Review AI recommendations vs your actions
   - Analyze hand histories for patterns
   - Update strategy based on results

3. **Long-term Tracking**
   - Maintain detailed records
   - Track improvement over time
   - Adjust approach based on trends

## Common Mistakes to Avoid

### Technical Mistakes

1. **Poor Calibration**
   - Rushing through calibration process
   - Not verifying detection accuracy
   - Using on unsupported game variants

2. **Incorrect Configuration**
   - Wrong monitor/region settings
   - Overly aggressive performance settings
   - Ignoring system capability warnings

### Strategic Mistakes

1. **Overriding AI Without Reason**
   - Ignoring recommendations due to "feelings"
   - Making emotional decisions
   - Not understanding the math behind recommendations

2. **Poor Bankroll Management**
   - Playing stakes too high for bankroll
   - Not setting or respecting stop-losses
   - Chasing losses after reaching limits

3. **Fatigue-Related Errors**
   - Playing too long without breaks
   - Making decisions when tired
   - Ignoring obvious mistakes due to fatigue

## Scenario-Based Quick Reference

### Quick Decision Guide

| Situation | Poker Action | Blackjack Action |
|-----------|-------------|------------------|
| **Premium Hand** | Raise 3-4x | Follow basic strategy |
| **Drawing Hand** | Call profitable odds | Hit to improve |
| **Weak Hand** | Fold to bets | Fold if negative EV |
| **Strong Position** | Play wider range | Increase bet size |
| **Short Stack** | Shove or fold | All-in strong hands |
| **Deep Stack** | Play post-flop | Follow deviations |

### Count-Based Adjustments

| True Count | Blackjack Adjustment | Poker Adjustment |
|------------|-------------------|------------------|
| **+4 or higher** | Max bet size, take insurance | Widen value ranges |
| **+2 to +3** | Increase bets, stand deviations | Slightly wider ranges |
| **0 to +1** | Normal play | Standard ranges |
| **-1 to -2** | Minimum bets | Slightly tighter ranges |
| **-3 or lower** | Minimum bets, hit deviations | Very tight ranges |

## Emergency Procedures

### Technical Emergencies

1. **Application Crash**
   ```bash
   # Restart application
   python main.py --calibrate  # If regions lost
   ```

2. **Calibration Loss**
   - Restart with `--calibrate` flag
   - Re-click all game regions carefully
   - Test detection before resuming play

3. **System Issues**
   - Save session data if possible
   - Restart computer if necessary
   - Check system resources (CPU, memory)

### Gaming Emergencies

1. **Casino Detection Risk**
   - Immediately stop using AI
   - Close overlay completely
   - Play manually for remainder of session
   - Consider changing tables/casinos

2. **Major Losing Streak**
   - Take mandatory break (30+ minutes)
   - Review recent decisions for patterns
   - Reduce bet sizes significantly
   - Consider ending session early

3. **Bankroll Emergency**
   - Immediately stop playing
   - Move to play money or free games
   - Review bankroll management rules
   - Take time off before returning

This usage guide provides comprehensive practical examples for effectively using the AI Blackjack Poker Assistant in real gaming scenarios. Always remember to use this tool responsibly and within your personal limits.