"""Command-line entrance; notebook and terminal share the same experiments."""
import argparse
import importlib
import json
from pathlib import Path
import platform
import sys
ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['doctor','lab','all'])
    p.add_argument('lab',nargs='?',type=int,default=1)
    p.add_argument('--lang',choices=['en','fr'],default='fr')
    p.add_argument('--output-dir')
    for name in ['neighbors','depth','clusters','min-leaf','width','epochs']:p.add_argument('--'+name,type=int)
    for name in ['slope','intercept','rate','temperature']:p.add_argument('--'+name,type=float)
    args=p.parse_args()
    if args.command=='doctor':
        versions={};problems={}
        for name in ['numpy','scipy','sklearn','matplotlib']:
            try:
                # ml_labs configures the writable Matplotlib cache before import.
                import os
                os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
                versions[name]=importlib.import_module(name).__version__
            except Exception as exc:problems[name]=type(exc).__name__
        print(json.dumps({'python':platform.python_version(),'executable':sys.executable,'folder':str(ROOT),'packages':versions,'setup_items':problems,'status':'READY' if not problems else 'INSTALL_CORE'},indent=2))
        return
    from ml_labs import run_lab
    options={k:v for k,v in vars(args).items() if k not in ['command','lab','lang','output_dir'] and v is not None}
    for number in (range(1,9) if args.command=='all' else [args.lab]):
        dest=Path(args.output_dir) / ('lab%02d'%number) if args.output_dir else None
        result=run_lab(number,args.lang,dest,**options)
        print(json.dumps(result['summary'],ensure_ascii=False,indent=2))
        print(result['report_path'])

if __name__=='__main__':
    try:main()
    except (ImportError,OSError,ValueError) as exc:
        print('SETUP / PARAMETER: '+str(exc))
        print('Open START_HERE.html / Ouvrir START_HERE.html')
        sys.exit(1)
