"""Read-only release checks for the standalone complementary study."""
from pathlib import Path
import hashlib,json,argparse
import pandas as pd
from docx import Document
ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--fresh',action='store_true');ap.add_argument('--write-result',action='store_true');a=ap.parse_args()
    lock=json.loads((ROOT/'protocol/source_lock.json').read_text())
    for name,digest in lock['sha256'].items():assert sha(ROOT/name)==digest,name
    original=json.loads((ROOT/'provenance/source_lock.json').read_text())['sha256']
    for name in ['src/simulation.py','src/eqtl.py','src/integration.py','src/run.py']:
        assert sha(ROOT/name)==original[name],name
    for name in ['diablo.R','check_R.R','check_HC3.R']:
        expected=(ROOT/'provenance/original_sources'/name).read_text(encoding='utf-8').replace('v3/','')
        assert (ROOT/'src'/name).read_text(encoding='utf-8')==expected,name
    amendment=json.loads((ROOT/'protocol/robustness_amendment.json').read_text())
    assert sha(ROOT/'src/robustness.py')==amendment['source_sha256']
    preserved=0
    if not a.fresh:
        manifest=json.loads((ROOT/'provenance/original_numerical_sha256.json').read_text())
        for name,digest in manifest.items():assert sha(ROOT/name)==digest,name;preserved+=1
    num=json.loads((ROOT/'results/numerical_audit.json').read_text())
    assert num['status']=='passed' and not num['failures']
    assert num['eqtl_cohorts']==1600 and num['integration_cohorts']==80 and num['integration_method_records']==560
    for name in ['R_validation','integration_replay','R_HC3_validation','robustness_audit','example_QC','complementary_audit']:
        assert json.loads((ROOT/f'results/{name}.json').read_text())['status']=='passed',name
    read=lambda n:pd.read_csv(ROOT/f'results/{n}.csv',keep_default_na=False,na_values=[''])
    t=read('mofa_discovery_tuning');m=read('mofa_discovery_examples');d=read('supervised_prediction_evaluation');s=read('diablo_signature_frequency')
    assert len(t)==80 and len(m)==8 and len(d)==400 and len(s)==288
    assert set(d.method)=={'DIABLO','Combined ridge','RNA ridge','Genotype ridge','Clinical only'}
    assert m.residual_programme_capture.between(0,1+1e-10).all()
    assert s.selection_frequency.between(0,1).all() and (s.replicates==10).all()
    assert len(list((ROOT/'figures').glob('*.png')))==13
    doc=Document(ROOT/'report/Genotype_RNA_eQTL_Calibration_Report.docx')
    assert doc.core_properties.author=='Vusumuzi Mabasa' and len(doc.inline_shapes)==13
    assert [len(x.rows) for x in doc.tables]==[4,5,5,9,5,13,5,3]
    assert len(doc.element.xpath('.//m:oMath'))>=18
    text=(ROOT/'README.md').read_text(encoding='utf-8')
    assert 'No real patient material' in text and 'post hoc' in text
    assert '## Results' in text and all(f'### Analysis {i}' in text for i in [1,2,3])
    assert not any(ord(c)<32 and c not in '\t\r\n' for c in text)
    investigation=ROOT/'investigations/eqtl_error'
    cal=json.loads((investigation/'results/investigation_audit.json').read_text())
    assert cal['status']=='passed' and cal['total_new_cohorts']==10400
    assert cal['seed_banks_disjoint'] and cal['source_locks_verified'] and cal['previous_results_unchanged']
    for line in (investigation/'checksums.sha256').read_text(encoding='utf-8-sig').splitlines():
        digest,name=line.split('  ',1);assert sha(investigation/name)==digest,name
    for name in ['independent_validation','paired_oracle_ablation','gaussian_pvalue_tails','conditional_interaction_null','strong_signal_power']:
        assert sha(ROOT/f'figures/calibration_{name}.png')==sha(investigation/f'figures/{name}.png'),name
    assert '4.25%' in text and '10,400' in text and '22/200' in text
    if not a.fresh:
        q=json.loads((ROOT/'results/report_qa.json').read_text())
        assert q['status']=='passed' and sha(ROOT/'README.md')==q['readme_sha256']
        assert sha(ROOT/'report/Genotype_RNA_eQTL_Calibration_Report.docx')==q['word_sha256']
        for line in (ROOT/'checksums.sha256').read_text().splitlines():
            digest,name=line.split('  ',1);assert sha(ROOT/name)==digest,name
    result=dict(status='passed',original_numerical_files_preserved=preserved,analyses=['MOFA molecular discovery','DIABLO supervised signatures','cis eQTL association'],original_eqtl_cohorts=1600,fresh_null_cohorts=800,additional_calibration_cohorts=10400,training_test_pairs=80,MOFA_descriptive_fixtures=8,supervised_method_records=400,figures=13,eqtl_calibration='Original excesses retained; independent confirmation did not reproduce persistent 11% F error. HC3 oracle excess retained. Wild bootstrap is a sensitivity candidate, not real-data certification.')
    if a.write_result:(ROOT/'results/release_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
