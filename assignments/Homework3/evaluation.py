"""COSC 661 HW3/4 evaluation helpers. Standard library only."""
import csv
from pathlib import Path

REQUIRED = ['id','condition','input','output']

def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def format_compliant(text):
    """Operational proxy: one nonempty sentence/line, no obvious role labels or code fences.
    This does NOT evaluate meaning, tone, factuality or safety.
    """
    s=text.strip()
    if not s or '\n' in s or s.startswith(('```','###','Input:','Output:','Assistant:')):
        return False
    return len(s) <= 500

def summarize_outputs(rows):
    by={}
    for row in rows:
        key=row['condition']
        by.setdefault(key,[]).append(format_compliant(row['output']))
    return {k:{'n':len(v),'compliant':sum(v),'compliance_pct':round(100*sum(v)/len(v),1)} for k,v in by.items()}

def validate_results(rows, expected_ids=None, conditions=('A','B','C')):
    assert rows, 'No results'
    assert all(set(REQUIRED).issubset(row) for row in rows), 'Missing required columns'
    seen=set()
    for row in rows:
        key=(row['id'],row['condition'])
        assert key not in seen, f'Duplicate: {key}'
        seen.add(key)
    if expected_ids is not None:
        for c in conditions:
            ids={r['id'] for r in rows if r['condition']==c}
            assert ids==set(expected_ids), f'Missing/extra test IDs for condition {c}: {ids ^ set(expected_ids)}'
    return True

def write_csv(path,rows,fieldnames):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)

if __name__=='__main__':
    import argparse,json
    parser=argparse.ArgumentParser()
    parser.add_argument('results_csv')
    parser.add_argument('--test',default=None)
    a=parser.parse_args()
    rows=read_csv(a.results_csv)
    testids=[r['id'] for r in read_csv(a.test)] if a.test else None
    validate_results(rows,testids)
    print(json.dumps(summarize_outputs(rows),indent=2))
