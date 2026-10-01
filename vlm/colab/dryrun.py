"""vlm/colab/dryrun.py — runs vlm/colab/APV_train_all.ipynb off Colab through nbclient, on the CPU with tiny steps
(APV_DRYRUN=1), against a small package made by `colab_pack.mjs --sample N`: the proof that every cell runs end to end.
The notebook is validated with nbformat first; --until <cell id> stops after that cell (the ids: settings, setup, gpu,
data, eye_train, eye_export, narrator_train, narrator_export, slots, zip). The executed notebook, with its outputs, is
written to <work>/executed.ipynb, and a summary per cell is printed.
  PYTHONPATH=<nbformat, nbclient, ipykernel> python vlm/colab/dryrun.py --data <package> --work <dir> [--until narrator_train]"""
import argparse, json, os, sys, time

NB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'APV_train_all.ipynb')
LACIE = '/Volumes/LaCie/astro-pilot/vlm'

def cells_until(nb, until, skip=()):
    ids = [c.get('id') for c in nb.cells]
    for x in [until, *skip]:
        if x and x not in ids: raise SystemExit(f'no cell {x} (ids {ids})')
    return [c for c in (nb.cells[:ids.index(until) + 1] if until else nb.cells) if c.get('id') not in skip]

def main(argv=None):
    import nbformat
    from nbclient import NotebookClient
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0]); ap.add_argument('--data', required=True); ap.add_argument('--work', required=True)
    ap.add_argument('--until', default=None); ap.add_argument('--skip', default='', help='cell ids to leave out, comma-separated'); ap.add_argument('--timeout', type=int, default=3600)
    ap.add_argument('--notebook', default=NB); a = ap.parse_args(argv)
    nb = nbformat.read(a.notebook, as_version=4); nbformat.validate(nb); nb.cells = cells_until(nb, a.until, [x for x in a.skip.split(',') if x])
    os.makedirs(a.work, exist_ok=True)
    os.environ.update({'APV_DRYRUN': '1', 'APV_DATA_DIR': a.data, 'APV_OUT_DIR': os.path.join(a.work, 'runs'), 'APV_LOCAL_DIR': os.path.join(a.work, 'local'),
                       'APV_ZIP_PATH': os.path.join(a.work, 'apv-models-v1.zip'), 'HF_HUB_OFFLINE': '1', 'HF_HOME': os.environ.get('HF_HOME', f'{LACIE}/hf'),
                       'SMOLVLM_DIR': os.environ.get('SMOLVLM_DIR', '/Volumes/LaCie/astro-pilot/test/out/vlm_explore/hfcache/HuggingFaceTB/SmolVLM-256M-Instruct'),
                       'TMPDIR': os.environ.get('TMPDIR') if os.environ.get('TMPDIR', '').startswith('/Volumes/LaCie/') else f'{LACIE}/tmp'})
    t0, client = time.time(), NotebookClient(nb, timeout=a.timeout, kernel_name='python3', resources={'metadata': {'path': a.work}}, allow_errors=False)
    err = None
    try: client.execute()
    except Exception as e: err = e
    nbformat.write(nb, os.path.join(a.work, 'executed.ipynb'))
    for c in nb.cells:
        if c.cell_type != 'code': continue
        text = ''.join(o.get('text', '') if o.get('output_type') == 'stream' else json.dumps(o.get('data', {}).get('text/plain', '')) if o.get('output_type') == 'execute_result' else f"{o.get('ename')}: {o.get('evalue')}" if o.get('output_type') == 'error' else '' for o in c.outputs)
        print(f"--- {c.get('id')} (execution {c.get('execution_count')}): {' | '.join(text.strip().splitlines()[-3:])[:600]}")
    print(f'dry run {"FAILED: " + str(err)[:2000] if err else "OK"} in {time.time() - t0:.0f} s; executed notebook: {os.path.join(a.work, "executed.ipynb")}')
    return 1 if err else 0

if __name__ == '__main__': sys.exit(main())
