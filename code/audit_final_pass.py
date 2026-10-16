"""Check final figures, retained statements/proofs, numeric preservation and PDF."""
import hashlib,json,re,zipfile
from pathlib import Path
import fitz
from build_final_figures import source_rows,DATASETS
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/artifacts/final_pass'

def numeric_cells(text):
    return [n for line in text.splitlines() if ' & ' in line
            for n in re.findall(r'\d*\.\d+|\d+',re.sub(r'\\rowcolor\{[^}]+\}','',line))]

def statements(text):
    return re.findall(r'\\begin\{(theorem|lemma|proposition|corollary)\}([\s\S]*?)\\end\{\1\}',text)

def main():
    proof_files=['app_order','app_views','app_lasso','app_perview','app_refit','app_generic','app_screening_extensions']
    checked=0;statement_count=0
    with zipfile.ZipFile(OUT/'before.zip') as z:
        for name in z.namelist():
            if name.startswith('tex/tables/'):
                assert numeric_cells(z.read(name).decode())==numeric_cells((ROOT/name).read_text()),name
                checked+=1
        for name in proof_files:
            path=f'tex/{name}.tex'
            assert z.read(path)==(ROOT/path).read_bytes(),path
        for name in ['theory','theory_generic','theory_practical_screen','theory_comparison']:
            path=f'tex/{name}.tex';old=statements(z.read(path).decode());new=statements((ROOT/path).read_text())
            assert old==new,path
            statement_count+=len(new)
        old_pdf=fitz.open(stream=z.read('main.pdf'),filetype='pdf')
        before=next(i for i,p in enumerate(old_pdf) if re.search(r'^References$',p.get_text(),re.M))
    S=source_rows();values=json.loads((OUT/'benchmark_plot_values.json').read_text())
    assert len(values)==24
    MAIN_DATASETS=[d for d in DATASETS if not d.startswith('Finance')]
    for q in values:
        if q['dataset'].startswith('Finance'):continue  # Finance comes from finance_failure.tex, not main tables
        j=MAIN_DATASETS.index(q['dataset']);rows=S[q['target']]
        assert tuple(q['ours'])==next(cs[j] for m,cs in rows if m=='MSWCSL')
        other=[(m,cs[j]) for m,cs in rows if m!='MSWCSL' and cs[j] is not None]
        if not other:assert q['strongest_completed'] is None
        else:
            name,vals=q['strongest_completed']
            assert tuple(vals)==next(cs[j] for m,cs in rows if m==name)
            assert vals[0]==max(v[0] for m,v in other)
    fmri={q['target']:q['ours'][0] for q in values if q['dataset']=='fMRI'}
    assert fmri=={'Both':.242,'Lagged':.416,'Summary':.428}
    mechanism=json.loads((OUT/'mechanism_plot_values.json').read_text())
    original=json.loads((ROOT/'experiments/results/independent_v5/summary.json').read_text())
    assert len(mechanism['persistence'])==32
    assert all(q in original['paired'] for q in mechanism['persistence'])
    assert len(mechanism['cancellation'])==90
    assert all(q in original['summary'] or q in original['matched'] for q in mechanism['cancellation'])
    log=(ROOT/'main.log').read_text()
    assert 'undefined' not in log.lower()
    assert 'Overfull' not in log
    assert 'multiply defined' not in log
    pdf=fitz.open(ROOT/'main.pdf')
    body=next(i for i,p in enumerate(pdf) if re.search(r'^References$',p.get_text(),re.M))
    assert body<=40,body
    # Inspect extracted text bounds, excluding top/bottom running furniture.
    out_of_bounds=[]
    for i,p in enumerate(pdf):
        for x0,y0,x1,y1,*_ in p.get_text('blocks'):
            if x0<0 or y0<0 or x1>p.rect.width+.5 or y1>p.rect.height+.5:out_of_bounds.append(i+1)
    assert not out_of_bounds,out_of_bounds
    report=dict(before_pages_excluding_references=before,pages_including_appendices_excluding_references=body,
        reference_pages=len(pdf)-body,total_pages=len(pdf),table_files_numerically_preserved=checked,
        proof_files_byte_identical=len(proof_files),formal_statements_unchanged=statement_count,
        benchmark_cells_checked=18,fmri_F1=fmri,paired_intervals_checked=32,cancellation_records_checked=90,
        undefined_references_or_citations=False,overfull_boxes=False,
        pdf_sha256=hashlib.sha256((ROOT/'main.pdf').read_bytes()).hexdigest())
    (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
