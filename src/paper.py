"""Shared research paper organised around three complementary questions."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SC=['null','shared_eqtl','subtype_eqtl','abundance_only']
SL=['Null','Shared eQTL','Subtype eQTL','Abundance only']


def table(head,rows):
    return '| '+' | '.join(head)+' |\n| '+' | '.join(['---']*len(head))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in rows)


def read(name):
    return pd.read_csv(ROOT/f'results/{name}.csv',keep_default_na=False,na_values=[''])


def main():
    old=(ROOT/'provenance/original_manuscript.txt').read_text(encoding='utf-8')
    generation=old.split('### Synthetic participants and local genetic structure',1)[1].split('### MOFA and supervised comparison',1)[0]
    qtlmethods=old.split('### Gene level cis eQTL tests',1)[1].split('## Results',1)[0]
    t=read('mofa_discovery_tuning');m=read('mofa_discovery_examples');p=read('supervised_prediction_summary');q=read('eqtl_summary');robust=read('robustness_summary');example=read('example_eqtl')
    qc=json.loads((ROOT/'results/example_QC.json').read_text())
    pred=lambda sc,method:float(p[(p.n==480)&(p.scenario==sc)&(p.method==method)].iloc[0].balanced_accuracy)
    roles=[['MOFA+','Discover molecular variation','Factors, loadings and reconstruction','Subtype accuracy does not define discovery success'],['DIABLO','Develop a supervised subtype signature','Independent prediction and feature reproducibility','A predictive feature is not necessarily an eQTL'],['cis eQTL','Test local genetic expression associations','Association, interaction and error control','Association does not establish causal mediation']]
    boundary=int((t.selected_k==6).sum())
    discovery=[]
    for sc,label in zip(SC,SL):
        a=t[(t.scenario==sc)&(t.n==480)];b=m[(m.scenario==sc)&(m.n==480)].iloc[0]
        discovery.append([label,f'{a.selected_k.median():.1f}',f'{a.masked_validation_mse.mean():.3f}',f'{b.genotype_projection_R2:.3f}',f'{b.RNA_projection_R2:.3f}',f'{b.residual_programme_capture:.3f}'])
    methods=['DIABLO','Combined ridge','RNA ridge','Genotype ridge','Clinical only']
    prediction=[[label]+[f'{pred(sc,method):.3f}' for method in methods] for sc,label in zip(SC,SL)]
    eqtl=[]
    for sc,label in zip(SC,SL):
        for family,name in [('omnibus','Any cis association'),('interaction','Subtype interaction')]:
            a=q[(q.n==480)&(q.scenario==sc)&(q.family==family)].iloc[0]
            eqtl.append([label,name,'Not applicable' if pd.isna(a.sensitivity) else f'{a.sensitivity:.3f}',f'{a.FDP:.3f}',f'{a.discoveries:.2f}'])
    negatives=[]
    for bank in ['original','fresh']:
        for method in ['Ordinary F','HC3 Wald F']:
            a=robust[(robust.bank==bank)&(robust.scenario=='abundance_only')&(robust.n==480)&(robust.method==method)].set_index('family')
            negatives.append([bank,method,f'{a.loc["omnibus","FDP"]:.3f}',f'{a.loc["interaction","FDP"]:.3f}'])
    genes=[[f'SIM GENE {int(a.gene)+1:02}',f'{a.omnibus_q:.3g}',f'{a.interaction_q:.3g}','Yes' if a.true_eqtl else 'No','Yes' if a.true_interaction else 'No'] for a in example.itertuples()]
    text=rf'''# Genotype and RNA Characterisation of Simulated Breast Cancer Subtypes

**Vusumuzi Mabasa**

**Simulation study.** All participants, genomic positions, LD, gene identifiers, RNA counts and TNBC labels are synthetic. The two molecular layers are germline genotype and RNA. No real patient material, breast cancer eQTL discoveries, protein or somatic assay layer is included. This is an illustrative design for Study 3, not a clinical or causal validation study.

[Word report](report/Genotype_RNA_eQTL_Report.docx) · [Quarto manuscript](report/manuscript.qmd) · [Example dataset](data/example) · [Analysis plan](protocol/complementary_analysis.json) · [Workflow](workflow.txt) · [Numerical provenance](provenance/packaging.json)

<!-- manuscript -->
## Abstract

We organised genotype and RNA analyses of simulated breast cancer cases around three complementary questions. MOFA+ described unsupervised molecular variation; DIABLO evaluated supervised TNBC versus non-TNBC signatures; and cis-eQTL models tested local genotype–expression associations and subtype interactions. The preserved simulation comprised 1,600 eQTL cohorts, 80 independent training/test pairs and 800 additional fresh negative-control cohorts at training sizes of 240 and 480. New post hoc summaries describe existing MOFA representations and DIABLO feature-selection reproducibility without refitting or changing the underlying simulations. In the subtype-eQTL scenario at N = 480, DIABLO balanced accuracy was {pred('subtype_eqtl','DIABLO'):.3f}, compared with {pred('subtype_eqtl','Combined ridge'):.3f} for a combined ridge reference and {pred('subtype_eqtl','RNA ridge'):.3f} for RNA ridge. These predictive results do not rank MOFA as a discovery method. The eQTL negative controls showed error excesses for ordinary F and HC3 references in some cells. The workflow illustrates how the three analyses can inform different aspects of Study 3 while retaining uncertainty, simple baselines and the need for further calibration and independent biological validation.

## Introduction

Matched germline genotype and RNA measurements can support several scientific questions in breast cancer cases. One concerns the major programmes of molecular variation; another concerns a reproducible signature of subtype; a third concerns genetic associations with expression. These questions overlap biologically but require different statistical objectives and evidence.

MOFA+ learns latent factors without using subtype labels during representation fitting [1,2]. DIABLO uses those labels to learn a multiblock signature [3,4]. A regional eQTL model tests genotype–expression associations rather than molecular clustering or classification [5,6]. The present study therefore does not define a MOFA+ versus DIABLO winner. Each method is interpreted against its intended question.

The study uses blood-like bulk RNA, idealised covariates and case-only TNBC labels. Synthetic genes deliberately avoid assigning invented associations to real breast cancer genes. No patient cohort, tissue-specific public eQTL dataset or actual breast cancer GWAS was analysed. The aim is to make a possible Study 3 analysis inspectable before real assay, cohort and validation requirements are finalised.

## Methods

### Three questions and their endpoints

Table 1 separates the objectives. MOFA discovery is not judged by DIABLO classification accuracy; DIABLO prediction is assessed against simpler prediction workflows; eQTL inference retains its own hypotheses and negative controls.

'''+table(['Analysis','Question','Evidence','Interpretive boundary'],roles)+r'''

![Three complementary analyses using the same molecular layers](figures/study_design.png)

Figure 1. Parallel analyses of genotype and RNA with distinct endpoints. The arrows indicate data inputs, not an established causal chain or a competition between methods.

### Simulation design and provenance

The two molecular panels contain 24 germline dosages across six synthetic local LD regions and 12 RNA genes. Four scenarios separate a molecular null, shared eQTLs, subtype-dependent eQTLs with abundance shifts, and abundance shifts without eQTLs. Sample sizes are 240 and 480. The eQTL experiment uses 200 cohorts per scenario and size; the integration experiment uses ten independently generated training cohorts per cell with 360 independent test cases each.

All original fitted-model, eQTL and robustness records are preserved byte for byte in this standalone repository. Numerical provenance identifies the source repository and commit. The scientific models were frozen before the original evaluation. Standalone packaging changes only file paths in the R scripts. The complementary framing, programme diagnostics and selection-frequency summaries were developed after the simulations; they are descriptive post hoc analyses, not newly preregistered endpoints. No effect sizes, seeds, model choices or test participants were selected or changed to support the reframing.

### Synthetic participants and local genetic structure

'''+generation+r'''

### Analysis 1 Molecular discovery with MOFA

The official mofapy2 engine fitted genotype and RNA views using Gaussian working likelihoods after training-only ancestry and batch adjustment and view standardisation. Gaussian dosage treatment is an approximation, not a binomial genotype likelihood. Subtype labels were excluded from preprocessing and factor fitting.

For each view m,

$$
X^{(m)}=ZW^{(m)\top}+E^{(m)}.
$$

Factor budgets 2, 4 and 6 were selected by 15% masked training entries in both views, with equal weight for the standardised view errors. Masked entries were excluded from preprocessing fits. A PCA start and a genuinely distinct perturbed-PCA start were selected by training ELBO. The selected model was refitted to the complete training views. Factor-budget counts and masked reconstruction are reported for all 80 fits. Selected validation errors are selection diagnostics, not independent-test reconstruction estimates.

Descriptive programme diagnostics use the eight already-saved first-replicate fixtures, one per scenario and sample size. The training projection reconstruction was summarised by

$$
R_m^2=1-\frac{\|X^{(m)}-\widehat ZW^{(m)\top}\|_F^2}{\|X^{(m)}-\overline X^{(m)}\|_F^2}.
$$

This uses plug-in Gaussian posterior-mean projections and is not the official MOFA posterior variance-decomposition output. It is a training reconstruction diagnostic; negative values are possible. Loadings and score plots describe the fitted representation without assigning real pathway names.

Capture of the three known residual programmes used centred orthonormal bases:

$$
S=\frac{\|Q_Z^\top Q_H\|_F^2}{3}.
$$

This quantity is invariant to sign changes and rotations. It measures only the planted residual H subspace, not every genetic, subtype or clinical contribution that MOFA may legitimately represent. It therefore cannot define total discovery success.

As an additional post hoc diagnostic, complete RNA was withheld for independent test cases. Fixed genotype loadings and precisions supplied genotype-only score projections:

$$
\widehat z_i^{(G)}=(I+W_G^\top\Lambda_GW_G)^{-1}W_G^\top\Lambda_Gx_i^{(G)},\qquad
\widehat x_i^{(R)}=W_R\widehat z_i^{(G)}.
$$

RNA error was compared with a training-mean baseline on the same standardised scale. These eight fixture-level results are supplied in the [discovery diagnostics table](results/mofa_discovery_examples.csv), without a claim to 80-replicate precision. Bootstrap factor stability, pathway enrichment and biological replication were not assessed.

### Analysis 2 Subtype signatures with DIABLO

Official R mixOmics DIABLO used genotype and RNA molecular blocks plus the same observed clinical information available to the prediction references. The clinical-input block is not an additional molecular assay. Threefold stratified training-only validation selected one or two components and four or eight features per block, bounded by available features. Preprocessing was repeated within each fold.

Weighted centroid voting produced subtype classes. Independent-test balanced accuracy was

$$
\mathrm{BA}=\frac{1}{2}\left\{\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}+
\frac{\mathrm{TN}}{\mathrm{TN}+\mathrm{FP}}\right\}.
$$

The relevant references are combined genotype/RNA ridge, RNA ridge, genotype ridge and clinical-only ridge classifiers, each with fixed C = 1, training-only scaling and the same independent test participants. Their fixed 0.5 probability threshold differs from DIABLO voting; results concern complete predictive workflows. Hard DIABLO labels were not treated as calibrated probabilities or used to create artificial ROC curves.

Molecular feature selection frequency was the proportion of ten independent simulation replicates in a cell in which a feature had a non-zero loading on any selected DIABLO component. It describes reproducibility across generated cohorts, not a bootstrap stability estimate for the same patients. The supervised response block was excluded from the feature summary. Constant-feature handling was checked before mapping feature indices back to original synthetic identifiers.

Previously computed MOFA and PCA classifier outputs remain in the inherited numerical records for provenance. They are not used as primary MOFA discovery endpoints or to rank MOFA against DIABLO in this report.

### Analysis 3 Local genotype expression associations

'''+qtlmethods+r'''

### Uncertainty and validation

Predictive means and approximate 95% Monte Carlo intervals use ten independent replicates per scenario and sample size. eQTL summaries use 200 cohorts per cell. These are simulation comparisons, not clinical confidence intervals or biological replication.

The original six model unit tests, complete integration replay, eight eQTL cohort replays, independent R test-statistic checks and probability/voted-class metric checks are retained. New validation checks orthonormal-subspace capture, fixed-model reconstruction arithmetic and the separation of discovery and supervised endpoints. The preservation manifest verifies all original fitted-model and eQTL records. Document pages are rendered and visually reviewed before publication.

## Results

### Analysis 1 Molecular representations and programme diagnostics

MOFA supplied an unsupervised representation rather than a subtype classifier ranking. Table 2 reports factor selection and reconstruction diagnostics at N = 480. Its selected budget and masked MSE summarise ten fits per cell; projection R2 and residual-programme capture describe only the fixed first-replicate fixture. Those scopes should not be combined into a fictitious repeated estimate.

Table 2. MOFA discovery diagnostics at N = 480. Training projection R2 and capture are descriptive fixture values, not posterior variance decomposition or clinical validation.

'''+table(['Scenario','Median K','Masked MSE','Genotype R2','RNA R2','Residual capture'],discovery)+r'''

The score and loading plots show how the model distributes variation among factors and molecular features. Subtype colours were added after unsupervised fitting. Apparent separation or overlap is descriptive; it is not the objective used to select the factor representation. The weak planted residual programmes are only one component of total molecular variation, so their capture should be interpreted alongside reconstruction and genetic structure.

![Unsupervised factors loadings and discovery diagnostics](figures/mofa_discovery.png)

Figure 2. MOFA scores and RNA loadings for the prespecified 240-case example, factor-budget counts across 80 fits and residual-programme capture in eight first-replicate fixtures. Capture points have no repeated-fit confidence intervals. Synthetic gene labels do not identify actual pathways.

### Analysis 2 Independent prediction and signature reproducibility

Table 3 evaluates DIABLO against prediction references. In the subtype-eQTL scenario, RNA-only and combined ridge performance show that much of the discriminating information is already available in RNA abundance. High accuracy does not establish that integration was necessary, that a selected feature is genetically regulated, or that the signature will transfer to clinical data. No MOFA discovery ranking follows from these results.

Table 3. Mean independent-test balanced accuracy at training N = 480 across ten simulation replicates.

'''+table(['Scenario','DIABLO','Combined ridge','RNA ridge','Genotype ridge','Clinical only'],prediction)+r'''

![Independent DIABLO prediction against simple references](figures/diablo_prediction.png)

Figure 3. Balanced accuracy with approximate 95% Monte Carlo intervals across ten independently generated training/test pairs per cell. MOFA classification is not part of this supervised benchmark. The dotted line is 0.5.

Feature-selection frequencies describe which RNA measurements repeatedly entered supervised signatures. Reproducible selection in these synthetic cohorts is useful as a workflow diagnostic, but it does not supply biological annotation or establish a regulatory mechanism. Ten replicates give frequencies only in increments of 0.1. In the N = 480 subtype-eQTL and abundance-only scenarios, the union across fitted components selected all twelve RNA genes in every replicate. This reflects limited selectivity in a small panel: a feature selected everywhere is not thereby established as a specific biomarker. Coefficient-direction stability was not assessed.

![RNA selection frequency in DIABLO signatures](figures/diablo_signature.png)

Figure 4. Proportion of ten independent N = 480 cohorts selecting each RNA feature on at least one fitted DIABLO component. This is simulation reproducibility, not a bootstrap estimate or external biomarker replication. Genotype feature frequencies are also released in the numerical table.

### Analysis 3 Cis associations and subtype interactions

Shared-eQTL cohorts contained six true cis-associated genes and no true interactions. Subtype-eQTL cohorts contained nine cis-associated genes and six interaction genes. Null and abundance-only cohorts contained no true genetic expression effects. Genetic mapping was evaluated against those planted truths independently of the classifier results.

Table 4. Gene-level sensitivity and realised mean false discovery proportion at N = 480 across 200 cohorts per scenario. Sensitivity is undefined where no true alternatives exist.

'''+table(['Scenario','Test family','Sensitivity','Mean FDP','Mean discoveries'],eqtl)+r'''

![Genetic association detection and false discoveries](figures/eqtl_performance.png)

Figure 5. Omnibus cis and subtype-interaction detection and error rates, with approximate 95% Monte Carlo intervals. Missing sensitivity points indicate settings without true alternatives. Strong planted effects and only twelve gene tests make these power estimates unsuitable for extrapolation to a transcriptome-wide clinical study.

The negative controls remain a substantive limitation. Ordinary F tests selected null genes in 11% of the original abundance-only N = 480 cohorts; HC3 reduced that point estimate to 8%. Fresh abundance-only values were 5.5% and 4.5%, respectively. HC3 also produced elevated complete-null rejection in other cells, including 8.5% at N = 240 and 9% at N = 480 in the fresh bank. Neither reference is certified for confirmatory eQTL use without further calibration.

Table 5. Original and fresh abundance-only negative controls at N = 480. All genes are null for genetic expression effects, so mean FDP equals the probability of selecting any gene.

'''+table(['Bank','Reference','Any cis gene','Any interaction gene'],negatives)+r'''

![Original and fresh eQTL negative controls](figures/eqtl_robustness.png)

Figure 6. Ordinary F and HC3 negative-control behaviour in original and fresh cohorts. Circles and squares distinguish the two banks. The original elevated rates are preserved; fresh results do not erase them or prove general calibration.

### A prespecified illustrative locus

The example dataset contains 240 simulated cases, including '''+str(qc['TNBC'])+r''' TNBC cases and '''+str(qc['non_TNBC'])+r''' non-TNBC cases. Its call rate is 100%, all 24 variants have valid dosage values, all twelve RNA measurements are finite, and the minimum observed minor allele frequency is '''+f'{qc["minimum_MAF"]:.3f}'+r'''. These properties reflect the idealised generator and do not replace real genotype or RNA-seq QC.

The first locus illustrates a deliberately planted subtype-dependent genotype–expression relationship. The local plot contains four synthetic variants and is not a genome-wide Manhattan plot. Correlated tag associations do not identify a causal nucleotide. This locus was fixed before evaluation rather than selected for an attractive signal.

![Prespecified genotype expression relationship](figures/example_eqtl.png)

Figure 7. Covariate-adjusted example expression by genotype dosage and subtype, and nominal local tag-association P values. SIM SNP 01 and SIM GENE 01 are invented identifiers; these are not findings about an actual breast cancer gene.

Table 6 retains every gene in the example, including non-significant genes and false discoveries. In particular, a significant interaction for a gene with a false truth label must not be reinterpreted as biological evidence.

Table 6. Original ordinary-F BH-adjusted P values and known simulation truth in the fixed example. Negative-control limitations apply.

'''+table(['Synthetic gene','Cis q','Interaction q','True cis effect','True interaction'],genes)+r'''

![Example association map with separate gene families](figures/example_qtl_map.png)

Figure 8. Omnibus and interaction q values for all twelve example genes; asterisks mark q below 0.05 within each separately adjusted family. Negative log10 q colours are capped at 15. This statistical map does not establish causal regulation.

## Discussion

The three analyses can inform different parts of Study 3 without competing for a single accuracy ranking. MOFA factors can generate hypotheses about molecular programmes and heterogeneity. DIABLO can test whether a supervised signature predicts subtype beyond simple references. eQTL models can identify candidate genotype–expression associations and possible subtype-dependent effects. A useful signature need not contain eQTL genes, and a shared eQTL need not distinguish subtypes.

The present discovery summaries are limited. MOFA is not guaranteed to recover weak residual programmes when substantial genotype, abundance and covariate variation is also present. The new reconstruction and capture diagnostics describe eight stored fixtures and are post hoc. Factor stability, actual pathway enrichment and independent programme replication remain unmeasured. A biological interpretation of real factors requires assay-aware annotation and validation rather than naming synthetic loadings as pathways.

DIABLO benefited from strong planted RNA abundance differences, and simpler RNA and combined ridge references performed well. The subtype-eQTL scenario also changes abundance, so its prediction results do not isolate the contribution of genetic interactions. Signature selection frequency across ten synthetic cohorts is an initial reproducibility check, not evidence of a clinically stable panel.

The eQTL analysis has a different weakness: some negative-control error rates exceed the nominal level. Robust covariance estimates do not solve every mean-model problem or finite-sample distortion. Further calibration, covariate sensitivity and larger fresh null banks are needed before confirmatory use. The small gene panel, common variants, synthetic LD and strong effects do not yield a definitive PhD power calculation.

All measurements are blood-like and synthetic. Real Study 3 work must address ancestry, relatedness, genotype uncertainty, expression QC, cell-composition estimation, clinical timing, case selection and independent replication. Blood associations do not establish tumour origin. Genetic association does not establish a causal allele, RNA mediation or breast cancer susceptibility. Fine-mapping, colocalisation with real GWAS and causal mediation require their own data and assumptions; they were not performed here.

## Conclusion

A genotype-and-RNA study can combine unsupervised molecular discovery, supervised subtype signatures and local genetic association mapping as complementary analyses. The current simulation illustrates that structure with preserved numerical results, purpose-specific figures and explicit limits. It does not define a MOFA+ versus DIABLO winner, identify real breast cancer eQTLs or provide a clinically validated biomarker.

## Reproducibility

The standalone repository contains the original simulation and fitted-model records, preserved numerical checksums, a small example dataset, purpose-specific analysis summaries, figures and a Quarto Word report authored by Vusumuzi Mabasa. Previously computed MOFA classifier records remain available for transparency but are not primary discovery endpoints.

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m unittest discover -s tests -v
python src/run.py --stage eqtl --workers 2
python src/run.py --stage integration --workers 2
Rscript --vanilla src/diablo.R .
python src/replay_integration.py
python src/analysis.py
Rscript --vanilla src/check_R.R .
python src/robustness.py
Rscript --vanilla src/check_HC3.R .
python src/complementary.py
python src/figures.py
python src/manuscript.py
python src/build_word.py
python src/audit.py --fresh
~~~

Existing records are resumed. Fresh generation requires a new output directory; see workflow.txt and the reproduction wrapper. Python dependencies are pinned. R with mixOmics and jsonlite and Quarto are required. Original source locks are archived alongside a documented packaging adjustment; no scientific model choices changed during relocation. Numerical audits pass independently of the scientific error-control limitations, which remain explicit.

## References

1. Argelaguet R, et al. MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data. Genome Biology. 2020;21:111. [Methods paper](https://doi.org/10.1186/s13059-020-02015-1).
2. MOFA developers. [Official preprocessing and interpretation guidance](https://biofam.github.io/MOFA2/faq.html).
3. Singh A, et al. DIABLO: an integrative approach for identifying key molecular drivers from multi-omics assays. Bioinformatics. 2019;35:3055–3062. [Methods paper](https://doi.org/10.1093/bioinformatics/bty1054).
4. mixOmics developers. [Official DIABLO documentation](https://mixomics.org/mixdiablo/).
5. GTEx Consortium. The GTEx Consortium atlas of genetic regulatory effects across human tissues. Science. 2020;369:1318–1330. [Research article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7737656/).
6. EMBL-EBI. [eQTL Catalogue](https://www.ebi.ac.uk/eqtl/).

## Data and code availability

All observations and genomic annotations in this study are synthetic. No patient records, private proposal documents or controlled-access data are included. The standalone study is available in [Breast Cancer Genotype RNA eQTL](https://github.com/Vusumuzi2Mabasa/Breast-Cancer-Genotype-RNA-eQTL) under the MIT licence. The [original repository](https://github.com/Vusumuzi2Mabasa/Breast-Multiomics-Copula-MOFA/tree/f3a2041) remains preserved.
'''
    text=text.replace('The score and loading plots show',f'The upper factor budget of six was selected in {boundary} of 80 fits. This boundary selection limits conclusions about an adequate factor dimension; larger budgets were not evaluated in this reframing. The score and loading plots show')
    (ROOT/'README.md').write_text(text,encoding='utf-8',newline='\n')
    print('Three complementary analyses research manuscript generated')


if __name__=='__main__':main()
