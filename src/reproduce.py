"""Reproduce the genotype/RNA experiment in a NEW directory."""
from pathlib import Path
import argparse,subprocess,shutil,sys,os
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--rscript',default=shutil.which('Rscript'));ap.add_argument('--quarto',default=shutil.which('quarto'));a=ap.parse_args()
    if not a.rscript or not a.quarto:ap.error('Rscript with mixOmics/jsonlite and Quarto required')
    target=a.output.resolve()
    if target.exists():ap.error('Choose a new output directory; released results are never deleted')
    target.mkdir(parents=True)
    for name in ['src','tests','protocol','provenance']:shutil.copytree(BASE/name,target/name,ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['requirements.txt','requirements-lock.txt','LICENSE']:shutil.copy2(BASE/name,target/name)
    for name in ['work','results','figures','report','data/example']:(target/name).mkdir(parents=True,exist_ok=True)
    investigation=target/'investigations/eqtl_error'
    investigation.mkdir(parents=True)
    shutil.copytree(BASE/'investigations/eqtl_error/src',investigation/'src',ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['results','figures']:(investigation/name).mkdir()
    for name in ['protocol.json','source_lock.json','conditional_null_amendment.json']:
        shutil.copy2(BASE/'investigations/eqtl_error'/name,investigation/name)
    shutil.copy2(BASE/'report/Genotype_RNA_eQTL_Report.docx',target/'report/Genotype_RNA_eQTL_Report.docx')
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
    def run(cmd):subprocess.run(cmd,cwd=target,env=env,check=True)
    run([sys.executable,'-m','unittest','discover','-s','tests','-v'])
    for stage in ['eqtl','integration']:run([sys.executable,'src/run.py','--stage',stage,'--workers',str(a.workers)])
    run([a.rscript,'--vanilla','src/diablo.R','.'])
    run([sys.executable,'src/replay_integration.py'])
    run([sys.executable,'src/analysis.py']);run([a.rscript,'--vanilla','src/check_R.R','.'])
    run([sys.executable,'src/robustness.py']);run([a.rscript,'--vanilla','src/check_HC3.R','.'])
    for name in ['complementary','figures']:run([sys.executable,f'src/{name}.py'])
    run([sys.executable,'investigations/eqtl_error/src/verify.py'])
    run([a.rscript,'--vanilla','investigations/eqtl_error/src/check_R.R','.'])
    for phase in ['diagnostic','validation']:
        run([sys.executable,'investigations/eqtl_error/src/run.py','--phase',phase,'--workers',str(a.workers)])
    for name in ['conditional_null','summarise','report']:
        run([sys.executable,f'investigations/eqtl_error/src/{name}.py'])
    for name in ['calibration_artifacts','manuscript']:run([sys.executable,f'src/{name}.py'])
    run([sys.executable,'src/build_word.py','--quarto',a.quarto]);run([sys.executable,'src/audit.py','--fresh'])
    print('Fresh scientific reproduction complete:',target)


if __name__=='__main__':main()
