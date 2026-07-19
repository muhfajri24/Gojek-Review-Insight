# Business Summary

## Scope

- Reviews analyzed: 6,162
- Sentiment distribution: {'negative': 2668, 'neutral': 349, 'positive': 3145}
- Labels are rating-derived: 1-2 negative, 3 neutral, and 4-5 positive.

## Model selection

- Selected model: Logistic Regression Balanced with basic preprocessing.
- Selection reason: highest test macro F1 (0.6327); weighted F1 was 0.8037.
- Best macro F1 by preprocessing experiment: {'basic': 0.6327476091299888, 'linguistic': 0.6144241372425746}.
- Hardest class by F1: neutral (0.2267).

## Evidence-backed finding

Finding:
The most frequent matched complaint category was **payment problems**.

Evidence:
It matched 576 negative reviews (21.59% of rating-derived negative reviews). Theme matches can overlap.

Recommendation:
Review representative examples and manually validate a sample before prioritizing product work. Use this category as a triage signal, not a causal diagnosis.

## Limitations

- Ratings are proxies for text sentiment and may disagree with review wording.
- Theme categorization uses transparent keyword rules, not topic modeling or human annotation.
- Theme categories overlap, so percentages do not sum to 100%.
- The default scope includes only app versions beginning with 4.8.
- Test results use one fixed grouped split and are not production-performance estimates.
