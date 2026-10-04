"""Execute every tracked notebook in a fresh kernel using this Python interpreter.

Default: write executed copies to build/notebooks. --write refreshes originals.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    paths=sorted(ROOT.glob('*.ipynb'))+sorted((ROOT/'Decision_trees').glob('*.ipynb'))
    if len(paths)!=11:raise RuntimeError(f'Expected 11 notebooks, found {len(paths)}')
    results=[]
    os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
    for path in paths:
        relative=path.relative_to(ROOT);print(f'RUN {relative}',flush=True)
        notebook=nbformat.read(path,as_version=4)
        for c in notebook.cells:
            if c.cell_type=='code':c.outputs=[];c.execution_count=None
        manager=KernelManager(kernel_name='python3')
        manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        NotebookClient(notebook,km=manager,timeout=180,allow_errors=False,
                       resources={'metadata':{'path':str(path.parent)}}).execute(cleanup_kc=True)
        target=path if args.write else ROOT/'build'/'notebooks'/relative
        target.parent.mkdir(parents=True,exist_ok=True);nbformat.write(notebook,target)
        results.append({'notebook':relative.as_posix(),'status':'passed'})
    output=ROOT/'build'/'notebook-checks.json';output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    print(f'PASS {len(results)} notebooks')

if __name__=='__main__':main()
