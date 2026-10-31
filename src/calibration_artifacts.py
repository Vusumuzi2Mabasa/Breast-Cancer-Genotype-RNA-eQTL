"""Copy the completed calibration figures and record investigation checksums."""
from pathlib import Path
import hashlib
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    investigation = ROOT / 'investigations/eqtl_error'
    for name in ['independent_validation', 'paired_oracle_ablation', 'gaussian_pvalue_tails', 'conditional_interaction_null', 'strong_signal_power']:
        for extension in ['png', 'svg']:
            shutil.copyfile(investigation / f'figures/{name}.{extension}', ROOT / f'figures/calibration_{name}.{extension}')
    rows = []
    for path in sorted(investigation.rglob('*')):
        if path.is_file() and path.name != 'checksums.sha256' and '__pycache__' not in path.parts:
            rows.append(hashlib.sha256(path.read_bytes()).hexdigest() + '  ' + path.relative_to(investigation).as_posix())
    (investigation / 'checksums.sha256').write_text('\n'.join(rows) + '\n', encoding='utf-8', newline='\n')
    print(f'Calibration figures copied; {len(rows)} investigation files hashed')


if __name__ == '__main__':
    main()
