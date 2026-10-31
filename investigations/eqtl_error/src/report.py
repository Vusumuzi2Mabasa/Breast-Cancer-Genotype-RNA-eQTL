"""Scientific plots and a self-contained investigation report with actual results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
FAMILIES=['omnibus','interaction'];METHODS=['Ordinary F','HC3 Wald F','Restricted HC2 wild F']
COLORS=['#2166ac','#b2182b','#1b7837']


def read(name):
    return pd.read_csv(ROOT/f'results/{name}.csv',keep_default_na=False,na_values=[''])


def save(fig,name):
    fig.savefig(ROOT/f'figures/{name}.png',dpi=180,bbox_inches='tight',facecolor='white')
    svg=ROOT/f'figures/{name}.svg';fig.savefig(svg,bbox_inches='tight',facecolor='white');plt.close(fig)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')


def main():
    summary=read('error_summary');pooled=read('pooled_confirmation');old=read('historical_null_cells');contrast=read('paired_contrasts');design=read('design_diagnostics')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(7.4,6),sharey=True)
    for row,n in enumerate([240,480]):
        for col,family in enumerate(FAMILIES):
            ax=axes[row,col]
            for j,(method,color) in enumerate(zip(METHODS,COLORS)):
                a=summary[(summary.phase=='validation')&(summary.n==n)&(summary.family==family)&(summary.method==method)].set_index('scenario').loc[['null','abundance_only']]
                ax.errorbar(np.arange(2)+(j-1)*.12,a.mean_FDP,yerr=[a.mean_FDP-a.lower,a.upper-a.mean_FDP],fmt='o',ms=4,capsize=2,color=color,label=method)
            ax.axhline(.05,color='grey',ls=':');ax.set(title=f'{family.capitalize()}, N={n}',xticks=[0,1],xticklabels=['Null','Abundance only'],ylabel='Any false gene after BH',ylim=(0,max(.16,float(summary[(summary.phase=='validation')&summary.full_null].upper.max())+.01)))
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.04),fontsize=8.5);fig.tight_layout();save(fig,'independent_validation')
    fig,ax=plt.subplots(figsize=(6.3,3.8))
    for j,(method,color) in enumerate(zip(METHODS,COLORS)):
        a=summary[(summary.phase=='conditional_null')&(summary.family=='interaction')&(summary.method==method)].set_index('n').loc[[240,480]]
        ax.errorbar(np.arange(2)+(j-1)*.12,a.mean_FDP,yerr=[a.mean_FDP-a.lower,a.upper-a.mean_FDP],fmt='o',capsize=2,ms=4,color=color,label=method)
    ax.axhline(.05,color='grey',ls=':');ax.set(title='Interaction null with common eQTL effects present',xticks=[0,1],xticklabels=['N=240','N=480'],ylabel='Any false interaction gene after BH',ylim=(0,.15));ax.legend(fontsize=8.5)
    fig.tight_layout();save(fig,'conditional_interaction_null')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.7),sharey=True)
    responses=['Observed log CPM plus one','Latent Gaussian','Log CPM without offset']
    for ax,method in zip(axes,METHODS[:2]):
        for j,(response,color) in enumerate(zip(responses,['#2166ac','#b2182b','#5e3c99'])):
            a=summary[(summary.phase=='diagnostic')&(summary.n==240)&(summary.family=='omnibus')&(summary.method==method)&(summary.response==response)].set_index('scenario').loc[['null','abundance_only']]
            ax.errorbar(np.arange(2)+(j-1)*.12,a.mean_FDP,yerr=[a.mean_FDP-a.lower,a.upper-a.mean_FDP],fmt='o',capsize=2,ms=4,color=color,label=response)
        ax.axhline(.05,color='grey',ls=':');ax.set(title=f'{method}, N=240',xticks=[0,1],xticklabels=['Null','Abundance only'],ylabel='Any false cis gene after BH',ylim=(0,.18))
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=1,bbox_to_anchor=(.5,1.22),fontsize=9);fig.tight_layout();save(fig,'paired_oracle_ablation')
    tails=read('pvalue_tails');fig,axes=plt.subplots(1,2,figsize=(7.4,3.6),sharey=True)
    for ax,n in zip(axes,[240,480]):
        for method,color in zip(METHODS[:2],COLORS[:2]):
            a=tails[(tails.phase=='diagnostic')&(tails.scenario=='null')&(tails.n==n)&(tails.response=='Latent Gaussian')&(tails.family=='omnibus')&(tails.method==method)].sort_values('threshold')
            ax.plot(a.threshold,a.rejection,'o-',ms=3,color=color,label=method)
        ax.plot([0,.05],[0,.05],':',color='grey');ax.set(title=f'Exact Gaussian response, N={n}',xlabel='Nominal gene P threshold',ylabel='Marginal rejection probability')
    axes[0].legend(fontsize=8);fig.tight_layout();save(fig,'gaussian_pvalue_tails')
    fig,axes=plt.subplots(1,2,figsize=(7.4,3.8),sharey=True)
    settings=[('shared_eqtl','omnibus'),('subtype_eqtl','omnibus'),('subtype_eqtl','interaction')]
    for ax,n in zip(axes,[240,480]):
        for j,(method,color) in enumerate(zip(METHODS,COLORS)):
            values=[summary[(summary.phase=='validation')&(summary.n==n)&(summary.scenario==sc)&(summary.family==family)&(summary.method==method)].iloc[0].sensitivity for sc,family in settings]
            ax.plot(np.arange(3)+(j-1)*.08,values,'o',color=color,label=method,ms=4)
        ax.set(title=f'Strong planted effects, N={n}',xticks=np.arange(3),xticklabels=['Shared cis','Subtype cis','Interaction'],ylabel='Mean true gene sensitivity',ylim=(0,1.03));ax.tick_params(axis='x',rotation=20)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,bbox_to_anchor=(.5,1.05),fontsize=8.5);fig.tight_layout();save(fig,'strong_signal_power')
    focal=old[(old.bank=='original')&(old.scenario=='abundance_only')&(old.n==480)&(old.method=='Ordinary F')&(old.family=='omnibus')].iloc[0]
    focal_fresh=pooled[(pooled.scenario=='abundance_only')&(pooled.n==480)&(pooled.method=='Ordinary F')&(pooled.family=='omnibus')].iloc[0]
    oracle=summary[(summary.phase=='diagnostic')&(summary.response=='Latent Gaussian')&(summary.scenario=='null')&(summary.n==240)&(summary.family=='omnibus')&(summary.method=='HC3 Wald F')].iloc[0]
    wild_null=summary[(summary.phase=='validation')&summary.scenario.isin(['null','abundance_only'])&(summary.method=='Restricted HC2 wild F')]
    lines=[
        'INVESTIGATION OF CIS eQTL ERROR RATES IN SIMULATED BREAST CANCER',
        'Author: Vusumuzi Mabasa',
        'All participants, genotypes, gene identifiers and expression values are synthetic.',
        '',
        'PRINCIPAL FINDINGS',
        'No arithmetic or generator mismatch was found in the checked implementations.',
        f'The original 11% ordinary-F cell did not reproduce as a stable rate: the same setting gave {focal_fresh.rate:.2%} across {int(focal_fresh.cohorts)} new cohorts (95% interval {focal_fresh.lower:.2%} to {focal_fresh.upper:.2%}).',
        f'HC3 showed a more reproducible problem with the eight-coefficient omnibus test: it rejected in {oracle.mean_FDP:.1%} of N240 complete-null cohorts even with the exact Gaussian oracle response.',
        f'The restricted wild candidate gave {wild_null.mean_FDP.min():.1%} to {wild_null.mean_FDP.max():.1%} rejection across the eight no-genetic-effect validation cells (null and abundance-only), with cell-specific uncertainty. Conditional interaction-null checks with common eQTL effects are reported separately. It is a promising sensitivity check, not a universal fix.',
        'Keep the original high-rate cell visible. The new evidence argues against a persistent 11% ordinary-F error rate for this generator; it does not prove that all ordinary F tests are calibrated for real RNA data.',
        '',
        'QUESTION',
        'Determine whether the original elevated false discoveries reflect a calculation error, persistent miscalibration, response-model assumptions, or Monte Carlo uncertainty.',
        '',
        'PRESERVATION AND IMPLEMENTATION CHECKS',
        'Original model sources, fitted-model records and published results were not modified.',
        'The generator was instrumented only to expose its exact latent Gaussian response; all original returned observations were checked for byte-equivalent numerical values in three pilot cohorts.',
        'Fast QR F and HC3 calculations replay the original implementation. Wild-bootstrap QR statistics agree with 93 naive regression refits and an independent R calculation.',
        '',
        'FRESH DESIGN',
        'Diagnostic bank: 4,000 cohorts (1,000 per null/abundance-only and N240/N480 cell), with paired observed, latent Gaussian and log-CPM-without-offset responses.',
        'Validation bank: 4,000 further null cohorts (1,000 per cell) and 400 strong-signal cohorts (100 per scenario/size cell). Its seeds are disjoint from the diagnostic and original banks.',
        'Additional conditional interaction-null bank: 2,000 fresh shared-eQTL cohorts (1,000 per sample size), with nonzero common genotype effects but no subtype interactions. This disclosed extension addresses the imprecision of the initial 100-cohort interaction check. Its seeds and method settings were fixed before those added cohorts were evaluated.',
        'A Windows worker-import issue occurred before any added conditional-null cohorts were generated. The launcher was corrected with a recorded repair; statistical methods and seeds did not change.',
        'Ordinary F and HC3 retain their original references. The candidate restricted wild test uses HC2-adjusted residuals from the reduced model, 999 participant-level Rademacher draws and plus-one P values.',
        'Multipliers are shared across genes/hypotheses within each cohort. BH 0.05 is applied separately across twelve omnibus and twelve interaction tests. No threshold is tuned from validation outcomes.',
        '',
        'WHAT THE ORIGINAL 11 PERCENT MEANS',
        f'The focal cell selected at least one null gene in {int(focal.rejections)} of {int(focal.cohorts)} cohorts ({focal.rate:.1%}); 95% Wilson interval {focal.lower:.1%} to {focal.upper:.1%}.',
        f'Its one-sided binomial inflation P value against 5% is {focal.inflation_p:.6g}; Holm adjustment across {len(old)} previously reported complete-null cells gives {focal.holm_p:.6g}. These historical tests are post hoc.',
        'When every gene is null, cohort-level any-discovery probability equals mean FDP/FDR. It is not the probability that an individual gene is significant.',
        '',
        'LARGER INDEPENDENT CONFIRMATION OF THE ORIGINAL REFERENCES',
        'The following pools only the two NEW, disjoint observed-response banks (2,000 cohorts per cell); it does not pool the selected original 200-cohort cell.',
    ]
    for a in pooled.itertuples():
        lines.append(f'{a.scenario}, N{a.n}, {a.method}, {a.family}: {a.rate:.1%} ({a.lower:.1%} to {a.upper:.1%}), {a.rejections}/{a.cohorts}.')
    lines+=['','PAIRED EXACT-GAUSSIAN ORACLE CHECK','The latent log2 abundance has a correctly specified homoscedastic Gaussian linear mean in these null settings. It is an unavailable oracle in real studies, used here to distinguish count-transformation effects from test-reference effects.']
    for a in summary[(summary.phase=='diagnostic')&(summary.response=='Latent Gaussian')&(summary.scenario=='null')].itertuples():
        lines.append(f'N{a.n}, {a.method}, {a.family}: {a.mean_FDP:.1%} ({a.lower:.1%} to {a.upper:.1%}).')
    lines+=['','CANDIDATE CORRECTION ON INDEPENDENT OBSERVED-RESPONSE COHORTS']
    for a in summary[(summary.phase=='validation')&(summary.method=='Restricted HC2 wild F')].itertuples():
        if a.full_null:lines.append(f'{a.scenario}, N{a.n}, {a.family}: false-discovery probability {a.mean_FDP:.1%} ({a.lower:.1%} to {a.upper:.1%}), {a.rejections}/{a.cohorts}.')
        else:lines.append(f'{a.scenario}, N{a.n}, {a.family}: mean FDP {a.mean_FDP:.3f}; true gene sensitivity {a.sensitivity:.1%} across {a.cohorts} strong-signal cohorts.')
    lines+=['','LARGER CONDITIONAL INTERACTION-NULL CONFIRMATION','Common eQTL effects are present, but every genotype-by-subtype interaction is null. Each cell below uses 1,000 newly seeded cohorts, not the initial 100-cohort power bank.']
    for a in summary[(summary.phase=='conditional_null')&(summary.family=='interaction')].itertuples():
        lines.append(f'N{a.n}, {a.method}: interaction false-discovery probability {a.mean_FDP:.1%} ({a.lower:.1%} to {a.upper:.1%}), {a.rejections}/{a.cohorts}.')
    lines+=['','INTERPRETATION AND LIMITS',
        'Interpret historical extremes using independent confirmation, not by replacing or silently discarding the original cell.',
        'Rejection under the exact Gaussian oracle cannot be attributed solely to transformed RNA counts. A difference between HC3 and ordinary F there implicates the finite-sample joint-Wald reference/covariance estimation, rather than an arithmetic coding discrepancy.',
        'The eight-coefficient omnibus HC3 test is more problematic than the four-coefficient interaction test. In the N240 Gaussian-null bank the paired HC3-minus-F omnibus rejection difference was 3.5 percentage points (cell-specific approximate 95% interval 1.9 to 5.1 points). This pattern supports investigating finite-sample joint-Wald inference, not abandoning all heteroskedasticity-robust methods.',
        'In that paired N240 null bank, observed versus Gaussian-oracle ordinary-F rejection differed by 0.4 percentage points (approximate interval -1.4 to 2.2 points). The response transformation was not the dominant detected explanation in this setting; this does not validate transformation choices for low-count real data.',
        'Leverage, condition numbers and subtype residual variance ratios are descriptive diagnostics; associations with rejection are not by themselves a causal decomposition.',
        'Wild bootstrap is an approximate candidate, not a distribution-free or universal calibration guarantee. Discrete Monte Carlo P values, estimated null models, BH dependence, small panels and case-only sampling remain relevant.',
        'Strong-signal power does not establish power for weak or realistic effects. Results apply to this generator and its 12-gene families, not a full transcriptome or real breast cancer cohort.',
        'Cell-wise confidence intervals and exploratory paired contrasts are not simultaneous guarantees. All original rates, response ablations, Bonferroni/BY diagnostics and unfavourable cells remain available.',
        '',
        'FILES',
        'protocol.json and source_lock.json: frozen fresh-bank design and sources.',
        'results/historical_null_cells.csv: historical rates, intervals and post hoc Holm tests.',
        'results/error_summary.csv: every diagnostic and validation cell.',
        'results/pooled_confirmation.csv: combined new observed-response null banks.',
        'results/paired_contrasts.csv: paired method and response differences.',
        'results/design_diagnostics.csv: leverage, rank, condition, residual variance and shape.',
        'results/pvalue_tails.csv: marginal P-value tail checks.',
        'results/investigation_audit.json: counts, disjoint seeds and source integrity.',
        '',
        'METHOD REFERENCE',
        'Davidson R, Flachaire E. The wild bootstrap, tamed at last. Journal of Econometrics. 2008. https://www.sciencedirect.com/science/article/pii/S0304407608000833',
        '',
        'REPRODUCTION',
        'Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS to 1.',
        'python investigations/eqtl_error/src/verify.py',
        'Rscript --vanilla investigations/eqtl_error/src/check_R.R .',
        'python investigations/eqtl_error/src/run.py --phase diagnostic --workers 2',
        'python investigations/eqtl_error/src/run.py --phase validation --workers 2',
        'python investigations/eqtl_error/src/conditional_null.py',
        'python investigations/eqtl_error/src/summarise.py',
        'python investigations/eqtl_error/src/report.py',
    ]
    (ROOT/'Investigation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    print('Investigation report and five scientific figures written')


if __name__=='__main__':main()
