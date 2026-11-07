| case | S | VarS | tau | Z | p | C | Cd | slope | lower_ci | upper_ci | prop_censored | fails now | fails after var-fix |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| simple_trend | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 0 | 0 |
| no_trend | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 0 | 0 |
| ties_annual | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ✅ | ❌→✅ | ✅ | 6 | 0 |
| all_identical | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌→✅ | ❌→✅ | ✅ | 10 | 8 |
| n2 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 10 | 10 |
| n3 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 10 | 10 |
| n4 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 10 | 10 |
| n5 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 0 | 0 |
| exact_linear | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 7 | 7 |
| lt_single_DL | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 7 | 0 |
| lt_multi_DL | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ✅ | ✅ | ✅ | 5 | 0 |
| gt_multi_AL | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ✅ | ❌→✅ | ✅ | 6 | 0 |
| mixed_cens | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ✅ | ❌→✅ | ✅ | 6 | 0 |
| monthly_nonseasonal | ✅ | ❌→✅ | ✅ | ❌→✅ | ✅ | ✅ | ✅ | ✅ | ❌→✅ | ✅ | ✅ | 3 | 0 |
| monthly_seasonal | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 0 | 0 |
| quarterly_seasonal | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 0 | 0 |
| seasonal_missing_month | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | 7 | 7 |
| seasonal_sparse_month | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | 9 | 9 |
| unequal_annual | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 6 | 0 |
| monthly_gaps | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 0 | 0 |
| multi_per_month_mid | ❌ | ❌→✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | 10 | 9 |
| multi_per_month_median | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | ✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 5 | 0 |
| multi_per_year_mid | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 6 | 0 |
| multi_per_year_median | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 6 | 0 |
| neg_and_zero | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 7 | 0 |
| with_nan | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | 0 | 0 |
| seasonal_lt | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ✅ | 9 | 9 |
| seasonal_gt | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 7 | 0 |
| long_6000 | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | 3 | 3 |
| real_ex1 | ✅ | ❌→✅ | ✅ | ❌→✅ | ✅ | ✅ | ✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 4 | 0 |
| real_ex2 | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ✅ | ✅ | 6 | 0 |
| real_ex3 | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 7 | 0 |
| real_ex4 | ✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ❌→✅ | ❌→✅ | ✅ | ❌→✅ | ❌→✅ | ✅ | 7 | 0 |

### Values (current code) for S, VarS, p, slope, CI
| case | metric | R | MannKS (current) | rel. diff | status | MannKS (var-fix) | status |
|---|---|---|---|---|---|---|---|
| simple_trend | S | 162 | 162 | 0.0e+00 | pass | 162 | pass |
| simple_trend | VarS | 950 | 950 | 0.0e+00 | pass | 950 | pass |
| simple_trend | p | 1.75544e-07 | 1.75544e-07 | 5.8e-10 | pass | 1.75544e-07 | pass |
| simple_trend | C | 1 | 1 | 1.1e-16 | pass | 1 | pass |
| simple_trend | slope | 0.517398 | 0.517398 | 2.1e-16 | pass | 0.517398 | pass |
| simple_trend | lower_ci | 0.454955 | 0.454955 | 9.8e-16 | pass | 0.454955 | pass |
| simple_trend | upper_ci | 0.590985 | 0.590985 | 3.8e-16 | pass | 0.590985 | pass |
| no_trend | S | -4 | -4 | 0.0e+00 | pass | -4 | pass |
| no_trend | VarS | 950 | 950 | 0.0e+00 | pass | 950 | pass |
| no_trend | p | 0.922462 | 0.922462 | 2.4e-16 | pass | 0.922462 | pass |
| no_trend | C | 0.538769 | 0.538769 | 2.1e-16 | pass | 0.538769 | pass |
| no_trend | slope | -0.00686557 | -0.00686557 | 3.0e-15 | pass | -0.00686557 | pass |
| no_trend | lower_ci | -0.0524201 | -0.0524201 | 1.1e-15 | pass | -0.0524201 | pass |
| no_trend | upper_ci | 0.04885 | 0.04885 | 7.1e-16 | pass | 0.04885 | pass |
| ties_annual | S | 110 | 110 | 0.0e+00 | pass | 110 | pass |
| ties_annual | VarS | 1481.33 | 1833.33 | 2.4e-01 | FAIL | 1481.33 | pass |
| ties_annual | p | 0.00462514 | 0.0109062 | 1.4e+00 | FAIL | 0.00462514 | pass |
| ties_annual | C | 0.997687 | 0.994547 | 3.1e-03 | FAIL | 0.997687 | pass |
| ties_annual | slope | 0.0465331 | 0.0465331 | 1.0e-15 | pass | 0.0465331 | pass |
| ties_annual | lower_ci | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| ties_annual | upper_ci | 0.0909034 | 0.0999863 | 1.0e-01 | FAIL | 0.0909034 | pass |
| all_identical | S | NA | 0 | inf | FAIL | 0 | FAIL |
| all_identical | VarS | NA | 408.333 | inf | FAIL | 0 | FAIL |
| all_identical | p | NA | 1 | inf | FAIL | 1 | FAIL |
| all_identical | C | NA | 0.5 | inf | FAIL | 0.5 | FAIL |
| all_identical | slope | NA | 0 | inf | FAIL | 0 | FAIL |
| all_identical | lower_ci | NA | 0 | inf | FAIL | NA | pass |
| all_identical | upper_ci | NA | 0 | inf | FAIL | NA | pass |
| n2 | S | NA | 1 | inf | FAIL | 1 | FAIL |
| n2 | VarS | NA | 1 | inf | FAIL | 1 | FAIL |
| n2 | p | NA | 1 | inf | FAIL | 1 | FAIL |
| n2 | C | NA | 0.5 | inf | FAIL | 0.5 | FAIL |
| n2 | slope | NA | 0.65745 | inf | FAIL | 0.65745 | FAIL |
| n2 | lower_ci | NA | 0.65745 | inf | FAIL | 0.65745 | FAIL |
| n2 | upper_ci | NA | 0.65745 | inf | FAIL | 0.65745 | FAIL |
| n3 | S | NA | 3 | inf | FAIL | 3 | FAIL |
| n3 | VarS | NA | 3.66667 | inf | FAIL | 3.66667 | FAIL |
| n3 | p | NA | 0.29627 | inf | FAIL | 0.29627 | FAIL |
| n3 | C | NA | 0.851865 | inf | FAIL | 0.851865 | FAIL |
| n3 | slope | NA | 1.04029 | inf | FAIL | 1.04029 | FAIL |
| n3 | lower_ci | NA | 0.684594 | inf | FAIL | 0.684594 | FAIL |
| n3 | upper_ci | NA | 1.39696 | inf | FAIL | 1.39696 | FAIL |
| n4 | S | NA | 6 | inf | FAIL | 6 | FAIL |
| n4 | VarS | NA | 8.66667 | inf | FAIL | 8.66667 | FAIL |
| n4 | p | NA | 0.0894294 | inf | FAIL | 0.0894294 | FAIL |
| n4 | C | NA | 0.955285 | inf | FAIL | 0.955285 | FAIL |
| n4 | slope | NA | 1.11944 | inf | FAIL | 1.11944 | FAIL |
| n4 | lower_ci | NA | 1.04371 | inf | FAIL | 1.04371 | FAIL |
| n4 | upper_ci | NA | 1.32044 | inf | FAIL | 1.32044 | FAIL |
| n5 | S | 10 | 10 | 0.0e+00 | pass | 10 | pass |
| n5 | VarS | 16.6667 | 16.6667 | 1.9e-15 | pass | 16.6667 | pass |
| n5 | p | 0.0274863 | 0.0274863 | 0.0e+00 | pass | 0.0274863 | pass |
| n5 | C | 0.986257 | 0.986257 | 1.1e-16 | pass | 0.986257 | pass |
| n5 | slope | 1.01884 | 1.01884 | 4.8e-15 | pass | 1.01884 | pass |
| n5 | lower_ci | 0.824404 | 0.824404 | 1.3e-16 | pass | 0.824404 | pass |
| n5 | upper_ci | 1.16094 | 1.16094 | 2.9e-15 | pass | 1.16094 | pass |
| exact_linear | S | NA | 66 | inf | FAIL | 66 | FAIL |
| exact_linear | VarS | NA | 212.667 | inf | FAIL | 212.667 | FAIL |
| exact_linear | p | NA | 8.30311e-06 | inf | FAIL | 8.30311e-06 | FAIL |
| exact_linear | C | NA | 0.999996 | inf | FAIL | 0.999996 | FAIL |
| exact_linear | slope | 0.5 | 0.5 | 1.1e-16 | pass | 0.5 | pass |
| exact_linear | lower_ci | 0.499886 | 0.499886 | 1.0e-15 | pass | 0.499886 | pass |
| exact_linear | upper_ci | 0.500068 | 0.500068 | 8.9e-16 | pass | 0.500068 | pass |
| lt_single_DL | S | 135 | 135 | 0.0e+00 | pass | 135 | pass |
| lt_single_DL | VarS | 1774 | 1812.33 | 2.2e-02 | FAIL | 1774 | pass |
| lt_single_DL | p | 0.00146529 | 0.00164591 | 1.2e-01 | FAIL | 0.00146529 | pass |
| lt_single_DL | C | 0.999267 | 0.999177 | 9.0e-05 | FAIL | 0.999267 | pass |
| lt_single_DL | slope | 0.0573131 | 0.0573131 | 0.0e+00 | pass | 0.0573131 | pass |
| lt_single_DL | lower_ci | 0.0313833 | 0.0310764 | 9.8e-03 | FAIL | 0.0313833 | pass |
| lt_single_DL | upper_ci | 0.0895436 | 0.0898265 | 3.2e-03 | FAIL | 0.0895436 | pass |
| lt_multi_DL | S | -45 | -45 | 0.0e+00 | pass | -45 | pass |
| lt_multi_DL | VarS | 2959.67 | 2999.67 | 1.4e-02 | FAIL | 2959.67 | pass |
| lt_multi_DL | p | 0.418641 | 0.42176 | 7.5e-03 | FAIL | 0.418641 | pass |
| lt_multi_DL | C | 0.79068 | 0.78912 | 2.0e-03 | FAIL | 0.79068 | pass |
| lt_multi_DL | slope | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| lt_multi_DL | lower_ci | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| lt_multi_DL | upper_ci | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| gt_multi_AL | S | 82 | 82 | 0.0e+00 | pass | 82 | pass |
| gt_multi_AL | VarS | 3113.33 | 3141.67 | 9.1e-03 | FAIL | 3113.33 | pass |
| gt_multi_AL | p | 0.146589 | 0.148423 | 1.3e-02 | FAIL | 0.146589 | pass |
| gt_multi_AL | C | 0.926705 | 0.925788 | 9.9e-04 | FAIL | 0.926705 | pass |
| gt_multi_AL | slope | 0.077383 | 0.077383 | 1.8e-16 | pass | 0.077383 | pass |
| gt_multi_AL | lower_ci | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| gt_multi_AL | upper_ci | 0.156559 | 0.156688 | 8.2e-04 | FAIL | 0.156559 | pass |
| mixed_cens | S | 99 | 99 | 0.0e+00 | pass | 99 | pass |
| mixed_cens | VarS | 3096.33 | 3140.67 | 1.4e-02 | FAIL | 3096.33 | pass |
| mixed_cens | p | 0.078209 | 0.080343 | 2.7e-02 | FAIL | 0.078209 | pass |
| mixed_cens | C | 0.960896 | 0.959829 | 1.1e-03 | FAIL | 0.960896 | pass |
| mixed_cens | slope | 0.0552963 | 0.0552963 | 1.3e-16 | pass | 0.0552963 | pass |
| mixed_cens | lower_ci | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| mixed_cens | upper_ci | 0.119998 | 0.120003 | 4.1e-05 | FAIL | 0.119998 | pass |
| monthly_nonseasonal | S | 3280 | 3280 | 0.0e+00 | pass | 3280 | pass |
| monthly_nonseasonal | VarS | 194365 | 194367 | 1.0e-05 | FAIL | 194365 | pass |
| monthly_nonseasonal | p | 1.02532e-13 | 1.02585e-13 | 5.3e-17 | pass | 1.02585e-13 | pass |
| monthly_nonseasonal | C | 1 | 1 | 3.3e-16 | pass | 1 | pass |
| monthly_nonseasonal | slope | 0.29634 | 0.29634 | 5.6e-16 | pass | 0.29634 | pass |
| monthly_nonseasonal | lower_ci | 0.244843 | 0.244843 | 1.0e-06 | FAIL | 0.244843 | pass |
| monthly_nonseasonal | upper_ci | 0.346378 | 0.346378 | 6.9e-07 | pass | 0.346378 | pass |
| monthly_seasonal | S | 250 | 250 | 0.0e+00 | pass | 250 | pass |
| monthly_seasonal | VarS | 1500 | 1500 | 0.0e+00 | pass | 1500 | pass |
| monthly_seasonal | p | 1.28317e-10 | 1.28317e-10 | 8.6e-07 | pass | 1.28317e-10 | pass |
| monthly_seasonal | C | 1 | 1 | 3.3e-16 | pass | 1 | pass |
| monthly_seasonal | slope | 0.203185 | 0.203185 | 1.1e-15 | pass | 0.203185 | pass |
| monthly_seasonal | lower_ci | 0.154123 | 0.154123 | 2.7e-15 | pass | 0.154123 | pass |
| monthly_seasonal | upper_ci | 0.24434 | 0.24434 | 6.8e-16 | pass | 0.24434 | pass |
| quarterly_seasonal | S | 150 | 150 | 0.0e+00 | pass | 150 | pass |
| quarterly_seasonal | VarS | 850.667 | 850.667 | 4.0e-16 | pass | 850.667 | pass |
| quarterly_seasonal | p | 3.2446e-07 | 3.2446e-07 | 3.2e-10 | pass | 3.2446e-07 | pass |
| quarterly_seasonal | C | 1 | 1 | 2.2e-16 | pass | 1 | pass |
| quarterly_seasonal | slope | 0.125529 | 0.125529 | 4.4e-16 | pass | 0.125529 | pass |
| quarterly_seasonal | lower_ci | 0.0880227 | 0.0880227 | 7.9e-16 | pass | 0.0880227 | pass |
| quarterly_seasonal | upper_ci | 0.167903 | 0.167903 | 2.0e-15 | pass | 0.167903 | pass |
| seasonal_missing_month | S | NA | 239 | inf | FAIL | 239 | FAIL |
| seasonal_missing_month | VarS | NA | 1375 | inf | FAIL | 1375 | FAIL |
| seasonal_missing_month | p | NA | 1.3773e-10 | inf | FAIL | 1.3773e-10 | FAIL |
| seasonal_missing_month | C | NA | 1 | inf | FAIL | 1 | FAIL |
| seasonal_missing_month | slope | 0.211265 | 0.211265 | 2.5e-15 | pass | 0.211265 | pass |
| seasonal_missing_month | lower_ci | 0.165842 | 0.165842 | 8.4e-16 | pass | 0.165842 | pass |
| seasonal_missing_month | upper_ci | 0.26032 | 0.26032 | 1.5e-15 | pass | 0.26032 | pass |
| seasonal_sparse_month | S | NA | 238 | inf | FAIL | 238 | FAIL |
| seasonal_sparse_month | VarS | NA | 1376 | inf | FAIL | 1376 | FAIL |
| seasonal_sparse_month | p | NA | 1.66874e-10 | inf | FAIL | 1.66874e-10 | FAIL |
| seasonal_sparse_month | C | NA | 1 | inf | FAIL | 1 | FAIL |
| seasonal_sparse_month | slope | 0.195752 | 0.195752 | 4.3e-16 | pass | 0.195752 | pass |
| seasonal_sparse_month | lower_ci | -2.83318 | 0.148562 | 1.1e+00 | FAIL | 0.148562 | FAIL |
| seasonal_sparse_month | upper_ci | 3.14916 | 0.242059 | 9.2e-01 | FAIL | 0.242059 | FAIL |
| unequal_annual | S | 153 | 153 | 0.0e+00 | pass | 153 | pass |
| unequal_annual | VarS | 1832.33 | 1833.33 | 5.5e-04 | FAIL | 1832.33 | pass |
| unequal_annual | p | 0.000383882 | 0.000385297 | 3.7e-03 | FAIL | 0.000383882 | pass |
| unequal_annual | C | 0.999808 | 0.999807 | 7.1e-07 | pass | 0.999808 | pass |
| unequal_annual | slope | 0.106623 | 0.106623 | 3.4e-15 | pass | 0.106623 | pass |
| unequal_annual | lower_ci | 0.0636545 | 0.0636414 | 2.1e-04 | FAIL | 0.0636545 | pass |
| unequal_annual | upper_ci | 0.13672 | 0.136753 | 2.4e-04 | FAIL | 0.13672 | pass |
| monthly_gaps | S | 413 | 413 | 0.0e+00 | pass | 413 | pass |
| monthly_gaps | VarS | 55800.3 | 55800.3 | 6.5e-16 | pass | 55800.3 | pass |
| monthly_gaps | p | 0.0811364 | 0.0811364 | 0.0e+00 | pass | 0.0811364 | pass |
| monthly_gaps | C | 0.959432 | 0.959432 | 3.5e-16 | pass | 0.959432 | pass |
| monthly_gaps | slope | 0.170997 | 0.170997 | 3.2e-16 | pass | 0.170997 | pass |
| monthly_gaps | lower_ci | 0.0123256 | 0.0123256 | 3.7e-15 | pass | 0.0123256 | pass |
| monthly_gaps | upper_ci | 0.302592 | 0.302592 | 1.8e-15 | pass | 0.302592 | pass |
| multi_per_month_mid | S | -549 | -545 | 7.3e-03 | FAIL | -545 | FAIL |
| multi_per_month_mid | VarS | 24580.3 | 24583.3 | 1.2e-04 | FAIL | 24580.3 | pass |
| multi_per_month_mid | p | 0.000473498 | 0.000521248 | 1.0e-01 | FAIL | 0.000520837 | FAIL |
| multi_per_month_mid | C | 0.999763 | 0.999739 | 2.4e-05 | FAIL | 0.99974 | FAIL |
| multi_per_month_mid | slope | -0.441489 | -0.432196 | 2.1e-02 | FAIL | -0.432196 | FAIL |
| multi_per_month_mid | lower_ci | -0.652761 | -0.646181 | 1.0e-02 | FAIL | -0.646173 | FAIL |
| multi_per_month_mid | upper_ci | -0.250066 | -0.244047 | 2.4e-02 | FAIL | -0.244079 | FAIL |
| multi_per_month_median | S | -716 | -716 | 0.0e+00 | pass | -716 | pass |
| multi_per_month_median | VarS | 24581.3 | 24583.3 | 8.1e-05 | FAIL | 24581.3 | pass |
| multi_per_month_median | p | 5.10553e-06 | 5.11004e-06 | 8.8e-04 | FAIL | 5.10553e-06 | pass |
| multi_per_month_median | C | 0.999997 | 0.999997 | 2.3e-09 | pass | 0.999997 | pass |
| multi_per_month_median | slope | -0.469505 | -0.469505 | 5.9e-16 | pass | -0.469505 | pass |
| multi_per_month_median | lower_ci | -0.623931 | -0.623934 | 4.4e-06 | FAIL | -0.623931 | pass |
| multi_per_month_median | upper_ci | -0.309356 | -0.309355 | 5.6e-06 | FAIL | -0.309356 | pass |
| multi_per_year_mid | S | 121 | 121 | 0.0e+00 | pass | 121 | pass |
| multi_per_year_mid | VarS | 949 | 950 | 1.1e-03 | FAIL | 949 | pass |
| multi_per_year_mid | p | 9.8051e-05 | 9.8884e-05 | 8.5e-03 | FAIL | 9.8051e-05 | pass |
| multi_per_year_mid | C | 0.999951 | 0.999951 | 4.2e-07 | pass | 0.999951 | pass |
| multi_per_year_mid | slope | 0.0907839 | 0.0907839 | 0.0e+00 | pass | 0.0907839 | pass |
| multi_per_year_mid | lower_ci | 0.0582643 | 0.0582629 | 2.4e-05 | FAIL | 0.0582643 | pass |
| multi_per_year_mid | upper_ci | 0.123192 | 0.123193 | 2.9e-06 | FAIL | 0.123192 | pass |
| multi_per_year_median | S | 129 | 129 | 0.0e+00 | pass | 129 | pass |
| multi_per_year_median | VarS | 949 | 950 | 1.1e-03 | FAIL | 949 | pass |
| multi_per_year_median | p | 3.25208e-05 | 3.28334e-05 | 9.6e-03 | FAIL | 3.25208e-05 | pass |
| multi_per_year_median | C | 0.999984 | 0.999984 | 1.6e-07 | pass | 0.999984 | pass |
| multi_per_year_median | slope | 0.0901697 | 0.0901697 | 3.1e-16 | pass | 0.0901697 | pass |
| multi_per_year_median | lower_ci | 0.0621328 | 0.0621072 | 4.1e-04 | FAIL | 0.0621328 | pass |
| multi_per_year_median | upper_ci | 0.112375 | 0.112401 | 2.3e-04 | FAIL | 0.112375 | pass |
| neg_and_zero | S | 104 | 104 | 0.0e+00 | pass | 104 | pass |
| neg_and_zero | VarS | 896.667 | 950 | 5.9e-02 | FAIL | 896.667 | pass |
| neg_and_zero | p | 0.00058234 | 0.000832486 | 4.3e-01 | FAIL | 0.00058234 | pass |
| neg_and_zero | C | 0.999709 | 0.999584 | 1.3e-04 | FAIL | 0.999709 | pass |
| neg_and_zero | slope | 0.200027 | 0.200027 | 2.8e-16 | pass | 0.200027 | pass |
| neg_and_zero | lower_ci | 0.11112 | 0.107245 | 3.5e-02 | FAIL | 0.11112 | pass |
| neg_and_zero | upper_ci | 0.2759 | 0.280523 | 1.7e-02 | FAIL | 0.2759 | pass |
| with_nan | S | 121 | 121 | 0.0e+00 | pass | 121 | pass |
| with_nan | VarS | 817 | 817 | 0.0e+00 | pass | 817 | pass |
| with_nan | p | 2.68964e-05 | 2.68964e-05 | 3.9e-12 | pass | 2.68964e-05 | pass |
| with_nan | C | 0.999987 | 0.999987 | 4.4e-16 | pass | 0.999987 | pass |
| with_nan | slope | 0.257275 | 0.257275 | 1.7e-15 | pass | 0.257275 | pass |
| with_nan | lower_ci | 0.202815 | 0.202815 | 1.6e-15 | pass | 0.202815 | pass |
| with_nan | upper_ci | 0.342877 | 0.342877 | 6.5e-16 | pass | 0.342877 | pass |
| seasonal_lt | S | NA | -65 | inf | FAIL | -65 | FAIL |
| seasonal_lt | VarS | NA | 1305 | inf | FAIL | 855.333 | FAIL |
| seasonal_lt | p | NA | 0.0764548 | inf | FAIL | 0.0286458 | FAIL |
| seasonal_lt | C | NA | 0.961773 | inf | FAIL | 0.985677 | FAIL |
| seasonal_lt | slope | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| seasonal_lt | lower_ci | -1.48201 | 0 | 1.0e+00 | FAIL | 0 | FAIL |
| seasonal_lt | upper_ci | 1.34492 | 0 | 1.0e+00 | FAIL | 0 | FAIL |
| seasonal_gt | S | 176 | 176 | 0.0e+00 | pass | 176 | pass |
| seasonal_gt | VarS | 1358.67 | 1500 | 1.0e-01 | FAIL | 1358.67 | pass |
| seasonal_gt | p | 2.05762e-06 | 6.2285e-06 | 2.0e+00 | FAIL | 2.05762e-06 | pass |
| seasonal_gt | C | 0.999999 | 0.999997 | 2.1e-06 | FAIL | 0.999999 | pass |
| seasonal_gt | slope | 0.157101 | 0.157101 | 2.8e-15 | pass | 0.157101 | pass |
| seasonal_gt | lower_ci | 0.0632768 | 0.0609894 | 3.6e-02 | FAIL | 0.0632768 | pass |
| seasonal_gt | upper_ci | 0.201954 | 0.203112 | 5.7e-03 | FAIL | 0.201954 | pass |
| long_6000 | S | 8.56082e+06 | 8.56082e+06 | 0.0e+00 | pass | 8.56082e+06 | pass |
| long_6000 | VarS | 2.4006e+10 | 2.4006e+10 | 2.0e-07 | pass | 2.4006e+10 | pass |
| long_6000 | p | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| long_6000 | C | 1 | 1 | 0.0e+00 | pass | 1 | pass |
| long_6000 | slope | 0.0061276 | 0.00613541 | 1.3e-03 | FAIL | 0.00613541 | FAIL |
| long_6000 | lower_ci | 0.00597702 | 0.00596967 | 1.2e-03 | FAIL | 0.00596967 | FAIL |
| long_6000 | upper_ci | 0.00627896 | 0.00628071 | 2.8e-04 | FAIL | 0.00628071 | FAIL |
| real_ex1 | S | 9723 | 9723 | 0.0e+00 | pass | 9723 | pass |
| real_ex1 | VarS | 1.64205e+06 | 1.64395e+06 | 1.2e-03 | FAIL | 1.64205e+06 | pass |
| real_ex1 | p | 3.2776e-14 | 3.39728e-14 | 1.2e-15 | pass | 3.28626e-14 | pass |
| real_ex1 | C | 1 | 1 | 1.0e-15 | pass | 1 | pass |
| real_ex1 | slope | 0.00089368 | 0.00089368 | 9.5e-14 | pass | 0.00089368 | pass |
| real_ex1 | lower_ci | 0.000729307 | 0.000729177 | 1.8e-04 | FAIL | 0.000729307 | pass |
| real_ex1 | upper_ci | 0.00105399 | 0.00105407 | 7.1e-05 | FAIL | 0.00105399 | pass |
| real_ex2 | S | -43 | -43 | 0.0e+00 | pass | -43 | pass |
| real_ex2 | VarS | 1020.33 | 1050.67 | 3.0e-02 | FAIL | 1020.33 | pass |
| real_ex2 | p | 0.188558 | 0.195066 | 3.5e-02 | FAIL | 0.188558 | pass |
| real_ex2 | C | 0.905721 | 0.902467 | 3.6e-03 | FAIL | 0.905721 | pass |
| real_ex2 | slope | -0.000166972 | -0.000166972 | 4.8e-13 | pass | -0.000166972 | pass |
| real_ex2 | lower_ci | -0.000500088 | -0.000500225 | 2.7e-04 | FAIL | -0.000500088 | pass |
| real_ex2 | upper_ci | 0 | 0 | 0.0e+00 | pass | 0 | pass |
| real_ex3 | S | 579 | 579 | 0.0e+00 | pass | 579 | pass |
| real_ex3 | VarS | 66957.7 | 67008.7 | 7.6e-04 | FAIL | 66957.7 | pass |
| real_ex3 | p | 0.0255019 | 0.0255579 | 2.2e-03 | FAIL | 0.0255019 | pass |
| real_ex3 | C | 0.987249 | 0.987221 | 2.8e-05 | FAIL | 0.987249 | pass |
| real_ex3 | slope | 0.00269331 | 0.00269331 | 1.5e-14 | pass | 0.00269331 | pass |
| real_ex3 | lower_ci | 0.000639939 | 0.000639882 | 8.9e-05 | FAIL | 0.000639939 | pass |
| real_ex3 | upper_ci | 0.00490727 | 0.00491004 | 5.6e-04 | FAIL | 0.00490727 | pass |
| real_ex4 | S | -73 | -73 | 0.0e+00 | pass | -73 | pass |
| real_ex4 | VarS | 586.333 | 589.333 | 5.1e-03 | FAIL | 586.333 | pass |
| real_ex4 | p | 0.00294475 | 0.00301829 | 2.5e-02 | FAIL | 0.00294475 | pass |
| real_ex4 | C | 0.998528 | 0.998491 | 3.7e-05 | FAIL | 0.998528 | pass |
| real_ex4 | slope | -0.976288 | -0.976288 | 5.7e-16 | pass | -0.976288 | pass |
| real_ex4 | lower_ci | -1.62253 | -1.62466 | 1.3e-03 | FAIL | -1.62253 | pass |
| real_ex4 | upper_ci | -0.579173 | -0.574341 | 8.3e-03 | FAIL | -0.579173 | pass |