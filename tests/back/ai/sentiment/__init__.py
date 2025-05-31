"""
Backend Sentiment Analysis AI Module Tests

This module defines unit tests for the backend sentiment analysis functionality, which takes text input and produces a numerical score between -1 and 1.

Test categories:
1. Unit tests for individual functions and classes in sentiment analysis components.
2. Accuracy tests validating that model output ranges match expected sentiment intervals.
3. Edge case tests for neutral sentiment, mixed sentiment, sarcasm, and domain-specific language.
4. Interval boundary tests ensuring correct interpretation of open and closed ranges.
5. Performance tests measuring throughput and latency in single-text and batch processing.

Expected outcomes are defined as mathematical intervals:
- (a;b) denotes an open interval excluding endpoints.
- <a;b> denotes a closed interval including endpoints.
- Mixed forms like <a;b) or (a;b> indicate mixed inclusion/exclusion.

Test data specification:
- testset.csv must contain:
    id: Unique test identifier prefixed with sa-<id>
    name: Human-readable name for the test
    description: Concise explanation of the test objective
    input_text: The text string to analyze
    expected_range: Sentiment interval using notation (a;b), <a;b>, (a;b> or <a;b>
    evaluation_method: "automated" or "manual" to indicate how results are verified
"""
