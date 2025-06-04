# Targeted Feedback System

## Overview

The CV Classification System now supports a sophisticated targeted feedback mechanism that allows users to provide specific feedback on individual classification results rather than general feedback on the entire classification.

## How It Works

### Structured Feedback Format

Users can provide feedback using the structured format:
```
area:: key:: feedback
```

Where:
- **area**: The classification area (`expertise`, `role_level`, or `org_unit`)
- **key**: The specific item being evaluated
- **feedback**: Your detailed feedback about that specific item

### Areas and Keys

#### Expertise Area
- **Area**: `expertise`
- **Keys**: Use the exact category name from the classification results
- **Example**: `expertise:: software_development:: This classification is very accurate`

#### Role Level Area
- **Area**: `role_level` 
- **Keys**: Can use:
  - Just the expertise name: `data_science`
  - Just the level name: `senior_level`
  - Combined format: `data_science-senior_level`
- **Example**: `role_level:: data_science-senior_level:: Should be mid_level instead`

#### Organizational Unit Area
- **Area**: `org_unit`
- **Keys**: Use the exact unit name from the classification results
- **Example**: `org_unit:: engineering:: Perfect match for this candidate`

### Multiple Feedback Items

You can provide feedback on multiple items by putting each on a separate line:

```
expertise:: software_development:: This classification is accurate
expertise:: marketing:: This should not be classified as marketing
role_level:: data_science-senior_level:: Level is too high, should be mid_level
org_unit:: engineering:: Excellent match
```

## Benefits

### Precision
- Feedback is applied only to the specific items you mention
- No more broad feedback affecting unrelated classifications
- More accurate learning from user input

### Learning
- The system builds targeted knowledge about specific categories, levels, and units
- Confidence adjustments are applied specifically to the items with feedback
- Future classifications of the same items are improved

### Transparency
- You can see exactly how your feedback affected specific classifications
- Clear indication of confidence adjustments based on your input
- Detailed justifications showing feedback influence

## Feedback Impact

### Confidence Adjustments
- **Positive feedback**: Increases confidence for that specific item in future classifications
- **Negative feedback**: Decreases confidence for that specific item
- **Mixed feedback**: No adjustment until a clear pattern emerges

### Adjustment Scaling
- More feedback = stronger adjustments
- Feedback strength plateaus at 5+ feedback entries per item
- Recent feedback has more weight than older feedback

### Learning Context
- Agents receive targeted feedback context in their prompts
- Specific examples of what users liked/disliked about each item
- Improved decision-making based on historical feedback patterns

## Backwards Compatibility

The system still supports the old general feedback format:
- If no structured format is detected, feedback is applied broadly
- Old feedback data remains intact and functional
- Users can mix structured and general feedback

## Usage Tips

1. **Be Specific**: Target your feedback to the exact items that need improvement
2. **Use Exact Keys**: Copy the exact category/unit names from the classification results
3. **Be Descriptive**: Provide clear explanations of why something is right or wrong
4. **Multiple Items**: Address multiple issues in one feedback session
5. **Check Validation**: The system shows real-time validation of your feedback format

## Examples

### Correcting Expertise Classification
```
expertise:: data_science:: This is correct, candidate has strong ML background
expertise:: marketing:: Should not be classified as marketing, no relevant experience
```

### Adjusting Role Levels
```
role_level:: software_development-senior_level:: Too high, candidate is mid-level
role_level:: project_management:: Missing this expertise area entirely
```

### Organizational Unit Feedback
```
org_unit:: engineering:: Perfect fit for this role
org_unit:: sales:: No sales experience evident in CV
```

### Mixed Feedback
```
expertise:: cybersecurity:: Accurate classification
role_level:: cybersecurity-entry_level:: Should be mid_level based on certifications
org_unit:: security:: Good match for security team
expertise:: web_development:: Missing this important skill area
```

## Technical Details

### Data Storage
- Feedback is stored with detailed metadata including timestamps and confidence levels
- Each feedback item is linked to specific classification results
- Historical feedback is preserved for trend analysis

### Processing
- Feedback is parsed and validated before storage
- Invalid formats are caught and reported to users
- Targeted feedback overrides general feedback when both are present

### Agent Integration
- All three agents (Expertise, Role Level, Org Unit) use targeted feedback
- Feedback context is included in agent prompts for better decision-making
- Confidence adjustments are applied during classification processing 