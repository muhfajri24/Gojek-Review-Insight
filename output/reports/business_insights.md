# Gojek Review Insight Summary

## Dataset Summary

- Total reviews after preprocessing: `6,152`
- Sentiment distribution: `{'positive': 3131, 'negative': 2670, 'neutral': 351}`
- Best model by F1-score: `Naive Bayes`
- Best F1-score: `0.8818`

## Main User Complaints

The most frequent terms in negative reviews point to themes such as: `driver, makin, gopay, mau, padahal, lama, bayar, malah`.
In practice, these terms usually relate to app performance issues, login friction, promotions, payments, or inconsistent ordering experiences.

## Positive Experience Drivers

Positive reviews more often highlight terms such as: `sangat, bantu, bagus, baik, mudah, mantap, cepat, banyak`.
This typically reflects appreciation for ease of use, service speed, attractive promotions, and the app's practical value in daily activities.

## Error Analysis

- Number of reviews misclassified by the best model: `140`
- Common failure patterns: mixed praise-and-complaint reviews, slang-heavy language, typos, and sarcastic phrasing.

## Business Recommendations

1. Prioritize investigation into the most dominant negative themes shown in `top_terms.csv` and `wordcloud_negative.png`.
2. Audit the user journey in the areas most frequently criticized, such as payments, promotions, and application stability.
3. Preserve the features users consistently praise and turn them into product communication strengths.
4. Use misclassified reviews as an additional source for expanding the Indonesian slang normalization dictionary.
