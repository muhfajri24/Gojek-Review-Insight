Place the Kaggle CSV file in this folder using the name `gojek_reviews.csv`.

The project will also try to read the first CSV file in this folder if the filename is different.

It is recommended to keep only the extracted CSV file here.
The downloaded Kaggle ZIP archive does not need to be stored in the repository.

Expected columns are review text (content or a supported alias) and either a
numeric rating (score or a supported alias) or an existing sentiment label.
The primary dataset uses ratings 1-5. The pipeline validates these values before
creating labels and reports a clear error for unsupported schemas or ratings.

The dataset is staged manually; the pipeline does not require Kaggle credentials.
Raw review data may contain usernames or other metadata. Do not publish or
redistribute it without checking the source license and privacy implications.
