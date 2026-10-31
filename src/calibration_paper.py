"""Integrate the completed calibration investigation without changing old results."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INV = ROOT / 'investigations/eqtl_error'


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)


def augment(text):
    text = text.replace('[Word report](report/Genotype_RNA_eQTL_Report.docx)', '[Updated Word report](report/Genotype_RNA_eQTL_Calibration_Report.docx) · [Original Word report](report/Genotype_RNA_eQTL_Report.docx)')
    summary = pd.read_csv(INV / 'results/error_summary.csv', keep_default_na=False, na_values=[''])
    pooled = pd.read_csv(INV / 'results/pooled_confirmation.csv', keep_default_na=False, na_values=[''])

    def one(frame, **conditions):
        chosen = frame
        for name, value in conditions.items():
            chosen = chosen[chosen[name] == value]
        assert len(chosen) == 1, conditions
        return chosen.iloc[0]

    def rate(row, column='any_false'):
        return f'{100*row[column]:.2f}% ({100*row.lower:.2f}–{100*row.upper:.2f})'

    methods = r'''### Independent investigation of eQTL calibration

Following the elevated historical negative-control rates, we generated 10,400 additional synthetic cohorts with seed banks disjoint from the original experiments. This investigation was motivated by the observed error excess; its historical comparisons are post hoc. The diagnostic and validation protocols and source hashes were frozen before their fresh evaluations. The extra conditional-null bank was specified after an imprecise 100-cohort interaction check and documented in an amendment before generating its 2,000 new cohorts. No threshold was selected to obtain a favourable result, and no original fitted result was replaced.

The diagnostic bank comprised 4,000 cohorts: 1,000 for each combination of the null or abundance-only scenario and N = 240 or 480. Ordinary F and HC3 tests were evaluated on the same cohorts using observed log2(CPM + 1), exact latent Gaussian log2 abundance and log2(CPM) without the offset. The latent response is an oracle available only because these data are simulated. Counts were positive in this panel; the no-offset diagnostic is not a recommendation for real low-count RNA.

The independent validation bank contained 4,000 no-genetic-effect cohorts plus 400 strong-signal cohorts. It evaluated both original tests and a restricted wild-bootstrap F candidate [7]. For each gene and hypothesis, the reduced-model fitted response and residuals defined

$$
Y_i^{*(b)}=\widehat{Y}_{0i}+\frac{\widehat{e}_{0i}}{\sqrt{1-h_{0ii}}}w_i^{(b)},\qquad
\Pr(w_i^{(b)}=1)=\Pr(w_i^{(b)}=-1)=\tfrac12.
$$

Here h is the reduced-model leverage; HC2 denotes the residual adjustment, not the HC3 covariance test. We refitted reduced and full models to each bootstrap response and compared their ordinary F statistic with the observed statistic. The same participant multipliers were shared across genes and hypotheses within each cohort. With B = 999 draws, the Monte Carlo P value was

$$
\widehat p=\frac{1+\sum_{b=1}^{B}I(F^{*(b)}\ge F_{\mathrm{obs}})}{B+1}.
$$

BH adjustment at 0.05 was applied separately to the twelve omnibus tests and twelve interaction tests. There is no joint error guarantee across the two families. In a wholly null family, the false discovery proportion is one whenever any gene is selected and zero otherwise; consequently its expectation equals the probability of at least one false discovery. For R independent cohorts, we estimated this probability by

$$
\widehat\pi_0=R^{-1}\sum_{r=1}^{R}I(V_r>0),
$$

where V is the number of false genes in a cohort. We report Wilson 95% intervals. Under partial alternatives we instead report mean FDP and sensitivity, because the probability of any false discovery is no longer the FDR. Intervals are cell-specific Monte Carlo intervals, not simultaneous guarantees.

The amended bank comprised 1,000 further shared-eQTL cohorts per sample size: common genotype effects were present but every subtype interaction was null. These cohorts assess interaction calibration in the presence of main genetic effects. Together, the three fresh banks comprise 10,400 cohorts; pooled ordinary-F and HC3 confirmation uses only the two fresh no-genetic-effect banks, giving 2,000 cohorts per scenario and size. The bootstrap estimates use the independent validation bank, not this pooled denominator.

Generator equivalence was checked against the original arrays and scalars. Original P values agreed within 3.1 × 10⁻¹⁴; 93 explicit bootstrap refits agreed within 2.8 × 10⁻¹⁴, and an independent R implementation agreed within 3.1 × 10⁻¹⁴. Bonferroni and BY adjustments, marginal P-value tails, leverage and condition numbers were retained as diagnostics. Arithmetic agreement alone does not establish statistical calibration.

'''
    rows = []
    for scenario, label in [('null', 'Null'), ('abundance_only', 'Abundance only')]:
        for n in [240, 480]:
            f = one(pooled, scenario=scenario, n=n, method='Ordinary F', family='omnibus')
            h = one(pooled, scenario=scenario, n=n, method='HC3 Wald F', family='omnibus')
            rows.append([label, n, rate(f, 'rate'), rate(h, 'rate')])
    confirmation = table(['Scenario', 'N', 'Ordinary F', 'HC3 Wald F'], rows)
    rows = []
    for n in [240, 480]:
        cells = [one(summary, phase='conditional_null', scenario='shared_eqtl', n=n, method=m, family='interaction') for m in ['Ordinary F', 'HC3 Wald F', 'Restricted HC2 wild F']]
        rows.append([n] + [rate(c) for c in cells])
    conditional = table(['N', 'Ordinary F', 'HC3 Wald F', 'Restricted wild F'], rows)
    results = r'''### Independent eQTL error rate investigation

The original 11% abundance-only omnibus error rate at N = 480 was not reproduced as a persistent rate. Across 2,000 fresh cohorts in that setting, ordinary F rejected at least one null gene in 85 cohorts, or 4.25% (95% interval 3.45–5.23%). The historical result remains 22/200, or 11% (7.4–16.1%); it is retained in Table 5 and Figure 6. The new evidence is consistent with an unusually high small-bank outcome, but does not prove that every original excess was due to chance or certify the model for real RNA.

Table 7. Omnibus probability of selecting any false gene after BH, with Wilson 95% intervals. Every cell uses 2,000 fresh cohorts pooled across the diagnostic and independent validation banks. All genes are null for genetic effects in these scenarios.

''' + confirmation + r'''

On the independent validation bank, ordinary-F estimates were broadly near the target whereas HC3 omnibus estimates remained elevated in several settings. The restricted wild bootstrap produced rates from 3.7% to 6.1% across the eight no-genetic-effect scenario, size and hypothesis cells. Every corresponding Wilson interval included 5%; this is compatibility with the target under this generator, not proof of exact control.

![Independent validation of eQTL error rates](figures/calibration_independent_validation.png)

Figure 9. Independent validation bank, with 1,000 fresh cohorts per null or abundance-only scenario and size. Intervals are cell-specific Wilson 95% intervals. These denominators differ from the pooled 2,000-cohort estimates in Table 7. All observations are simulated.

The paired oracle diagnostic separated response transformation from test-reference behaviour. For the Gaussian null at N = 240, ordinary-F omnibus rejection was 5.0%, whereas HC3 rejection was 8.5%. Their paired difference was 3.5 percentage points (approximate 95% interval 1.9–5.1). Observed-response versus oracle ordinary-F rejection differed by only 0.4 points (−1.4 to 2.2). Count transformation was therefore not the dominant detected explanation in this setting. Excess HC3 rejection under the Gaussian control supports a finite-sample joint-Wald calibration problem rather than an arithmetic implementation discrepancy.

![Paired oracle and observed response diagnostics](figures/calibration_paired_oracle_ablation.png)

Figure 10. Response diagnostics at N = 240 using identical simulated cohorts across response transformations and methods. Points show probabilities of any false cis gene after BH, with cell-specific Wilson 95% intervals. The paired differences are reported in the text and accompanying tables. The latent Gaussian response is an oracle unavailable in a real study.

HC3 excesses also appeared in the small marginal P-value tails and persisted under Bonferroni adjustment. This shows that the pattern cannot be explained solely by BH adjustment. The eight-coefficient omnibus test was more affected than the four-coefficient interaction test. Leverage and condition numbers were inspected, but these descriptive diagnostics do not identify a unique causal source of distortion.

![Gaussian oracle P value tail calibration](figures/calibration_gaussian_pvalue_tails.png)

Figure 11. Marginal P-value tail checks under the synthetic Gaussian null. Deviations concern the finite-sample test behaviour; they do not invalidate all heteroskedasticity-robust inference.

The initial independent power bank contained only 100 cohorts per signal cell and yielded an imprecise 8/100 bootstrap interaction-error estimate for shared eQTLs at N = 480. We therefore generated the amended 2,000-cohort bank rather than treating that small estimate as conclusive. With main effects present and all interactions null, bootstrap interaction rejection was 4.4% at N = 240 and 4.6% at N = 480. The original 8/100 check remains in the released tables.

Table 8. Interaction probability of any false gene after BH in the amended shared-eQTL bank, with Wilson 95% intervals; 1,000 new cohorts per size. Common genetic effects are present, but every interaction is null.

''' + conditional + r'''

![Interaction calibration with common genetic effects](figures/calibration_conditional_interaction_null.png)

Figure 12. Conditional interaction-null validation on new synthetic cohorts, separate from both the no-genetic-effect banks and the 100-cohort power checks. Cell-specific intervals include the 5% target for all displayed methods and sizes.

Strong planted signals were retained by the bootstrap candidate: mean omnibus sensitivity was 100% in the initial signal checks and subtype-interaction sensitivity was 98% at N = 240 and 100% at N = 480. These ceiling results do not establish power for weak effects, rare variants or a transcriptome-wide study. The shared-eQTL interaction family is null and is assessed for error rather than sensitivity.

![Strong signal sensitivity checks](figures/calibration_strong_signal_power.png)

Figure 13. Sensitivity on 100 independent synthetic cohorts per strong-signal scenario and size. Power estimates are conditional on the large planted effects; the investigation did not establish a weak-effect power advantage.

The completed investigation, all specified cells and implementation checks are available in the [investigation report](investigations/eqtl_error/Investigation.txt), [complete calibration tables](investigations/eqtl_error/results/error_summary.csv), [pooled confirmation](investigations/eqtl_error/results/pooled_confirmation.csv) and [source and seed audit](investigations/eqtl_error/results/investigation_audit.json). Original numerical results were preserved byte for byte.

'''
    text = text.replace('## Results\n', methods + '## Results\n', 1)
    text = text.replace('## Discussion\n', results + '## Discussion\n', 1)
    text = text.replace('The eQTL negative controls showed error excesses for ordinary F and HC3 references in some cells.', 'Historical eQTL negative controls showed error excesses in some cells. An additional 10,400-cohort investigation did not reproduce the original 11% ordinary-F rate as a persistent excess: the corresponding fresh estimate was 4.25% (95% interval 3.45–5.23%). HC3 remained elevated under Gaussian oracle controls; restricted wild-bootstrap checks were closer to nominal error, with limitations.')
    old = 'The eQTL analysis has a different weakness: some negative-control error rates exceed the nominal level. Robust covariance estimates do not solve every mean-model problem or finite-sample distortion. Further calibration, covariate sensitivity and larger fresh null banks are needed before confirmatory use. The small gene panel, common variants, synthetic LD and strong effects do not yield a definitive PhD power calculation.'
    new = 'The larger calibration investigation changes the interpretation of the historical eQTL error excess. Ordinary F remains a reasonable baseline under this generator, because its original 11% rate was not persistent across the larger fresh banks. HC3 should not be treated as an automatic correction: its omnibus rejection remained excessive under an exact Gaussian control. Restricted wild bootstrap is a promising sensitivity analysis, including for interactions with common eQTL effects present, but its validity remains design-dependent. These results do not justify silently replacing historical discoveries or retrospectively recalibrating selected genes. Real-data and weak-effect validation, covariate sensitivity, relatedness, low counts and transcriptome-wide multiplicity remain unresolved. The small gene panel, common variants, synthetic LD and strong effects do not yield a definitive PhD power calculation.'
    assert old in text
    text = text.replace(old, new)
    text = text.replace('The current simulation illustrates that structure with preserved numerical results, purpose-specific figures and explicit limits.', 'The current simulation illustrates that structure with preserved numerical results, purpose-specific figures and explicit limits. Independent calibration supports ordinary F as a simulation baseline and restricted wild bootstrap as a sensitivity analysis; it does not establish confirmatory eQTL validity for real patients.')
    text = text.replace('## References\n', 'The new calibration investigation is reproduced separately using the commands in [its report](investigations/eqtl_error/Investigation.txt). Its seed locks, amended conditional-null design, raw records, figures and checksums are retained under investigations/eqtl_error. The original study and the subsequent calibration banks must not be conflated.\n\n## References\n', 1)
    text = text.replace('## Data and code availability\n', '7. Davidson R, Flachaire E. The wild bootstrap, tamed at last. Journal of Econometrics. 2008;146:162–169. [Methods paper](https://doi.org/10.1016/j.jeconom.2008.08.003).\n\n## Data and code availability\n', 1)
    return text
