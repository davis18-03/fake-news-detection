# Data Schema for Fake News Detection

## Required CSV Files
- `Fake.csv`: Contains fake news articles
- `True.csv`: Contains real news articles

## Expected Columns
| Column | Type   | Description                  |
|--------|--------|------------------------------|
| text   | string | The news article text/content |

## Optional Columns (ignored by pipeline)
| Column   | Type   | Description                |
|----------|--------|----------------------------|
| title    | string | (Optional) Article title   |
| subject  | string | (Optional) News subject    |
| date     | string | (Optional) Date published  |

## Example (Fake.csv or True.csv)
| text                        | title           | subject | date       |
|-----------------------------|-----------------|---------|------------|
| Donald Trump is dead        | Trump Dead      | News    | 1/1/2020   |
| NASA launches new satellite | Space Progress  | Science | 2/2/2020   |

- Only the `text` column is required. Other columns will be ignored by the pipeline.
- Each file should contain one article per row. 