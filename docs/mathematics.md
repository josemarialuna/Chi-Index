# Mathematical definition and verification

Reference: Luna-Romera et al. (2019), DOI
[10.1016/j.ins.2019.02.046](https://doi.org/10.1016/j.ins.2019.02.046),
Section 3.4, equations (5)–(14).

Let N be the contingency table with r observed clusters (rows) and c observed
classes (columns). Build R by dividing each row by its total, and C by dividing
each column by its total. Multiply both tables by 100.

For each table T, compute Pearson's statistic:

```text
E[i,j] = row_total[i] * column_total[j] / grand_total
chi_squared(T) = sum((T[i,j] - E[i,j])**2 / E[i,j])
row_max = 100 * r * (min(r,c) - 1)
column_max = 100 * c * (min(r,c) - 1)
a = chi_squared(R) / row_max
b = chi_squared(C) / column_max
Chi Index = a + b - abs(a-b) = 2 * min(a,b)
```

We explicitly disable Yates' continuity correction. SciPy enables it by default
for tables with one degree of freedom; that correction is absent from the
paper's equations and would prevent a perfect 2-by-2 partition from scoring 2.
See [SciPy's documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html).
Only the statistic is used, not the p-value. This is a clustering score, not
a hypothesis test on the original sample counts.

The maximum score is 2. We clip floating-point roundoff to [0, 2]. The
implementation does not round intermediate percentages.

## Degenerate inputs

The published normalization divides by zero when r or c is 1. We explicitly
return 0 in these cases, including when both partitions are constant. This is
a library convention, not a value obtained by substituting into equations
(12) and (13). Empty inputs, missing labels and infinite numeric labels are
rejected. Unobserved categorical levels do not add rows or columns.

## Paper's worked example

From the integer counts in Table 4, applying Pearson without intermediate
rounding gives:

| k | row chi-squared | column chi-squared | row maximum | column maximum | score |
|---|---:|---:|---:|---:|---:|
| 2 | 90.476190 | 139.409499 | 200 | 300 | 0.904762 |
| 3 | 277.500000 | 299.380805 | 600 | 600 | 0.925000 |
| 4 | 304.093567 | 238.237516 | 800 | 600 | 0.760234 |

Table 5 prints scores 0.890, 0.925 and 0.760. For k=2, equation (3) uses rounded
percentages (13 and 37 instead of 12.5 and 37.5), explaining the different
row statistic. There are additional inconsistencies in the printed relative
tables: Table 4(e)'s third row is not the normalization of Table 4(b)'s third
row. The software uses the integer counts and full-precision equations, not
the rounded or inconsistent illustrative percentages. The best k in this
example remains 3.

## Interpretation limits

The index does not guarantee that its maximum occurs at the number of classes,
nor that a score of 2 uniquely identifies the ground-truth partition. For
example, counts [[2,0,0],[0,2,2]] reach 2 under the published normalization,
despite merging two classes. This follows from the equations; changing it would
define a different index. The model search selects the smallest k on an exact
tie and does not change the mathematical definition to force a particular k.

## Tests

Tests calculate Pearson independently with scalar loops, check Tables 2 and 4,
and exercise perfect agreement, independence, rectangular tables, random
tables, label renaming, paired permutations, symmetry and sample replication.
They also compare class exports against the standalone metric and verify that
the retained model and centroid plot belong to the selected k.

These checks validate this Python implementation. They do not reproduce the
paper's entire Spark benchmark or establish the absence of all possible bugs.
