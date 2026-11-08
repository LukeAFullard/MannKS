# V-31 Data Quality Analysis Notes Validation

This validation case explicitly triggers and verifies the data quality warnings defined in `analysis_notes.py`.

## Results
| Test ID                     | Expected Note                                                 | Actual Notes                                                                                                                   | Match   |     Slope |     P-Value |
|:----------------------------|:--------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|:--------|----------:|------------:|
| V-31_All_NA                 | Data all NA values                                            | Data all NA values                                                                                                             | True    | nan       | nan         |
| V-31_Low_Unique             | < 3 unique values                                             | < 3 unique values; < 3 unique values                                                                                           | True    |   0.125   |   0.014214  |
| V-31_Low_NonCensored        | < 5 Non-censored values                                       | < 5 Non-censored values; < 5 Non-censored values                                                                               | True    |  -2.00006 |   0.0678363 |
| V-31_Long_Run               | Long run of single value                                      |                                                                                                                                | False   |  -0.125   |   0.083798  |
| V-31_Sen_Slope_LeftCensored | influenced by left-censored values                            | Input data `x` contains strings. Please pre-process it with `prepare_censored_data` first.                                     | False   | nan       | nan         |
| V-31_Sen_Slope_Tied         | based on tied non-censored values                             | < 3 unique values; sample size (5) below minimum (10); < 3 unique values; WARNING: Sen slope based on tied non-censored values | True    |   0       |   0.288844  |
| V-31_Critical_Censored      | CRITICAL: Sen slope is based on a pair of two censored values | Input data `x` contains strings. Please pre-process it with `prepare_censored_data` first.                                     | False   | nan       | nan         |