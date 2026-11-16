"""Run training conditions in parallel, one per GPU, and wait for them (aerial-vg).

Used by the Kaggle notebook. Each job is `bash scripts/run_one.sh <name> <geo_aug> <gpu>` started with
subprocess.Popen and waited on directly (no process-list polling), with unbuffered logs in logs/<name>.log.
If a job fails, the others are stopped and the log tails are printed, so a broken run fails in minutes,
not at the 12 h limit.

  python scripts/launch.py baseline:hflip:0 d4_remap:d4_remap:1
"""
import os
import signal
import subprocess
import sys
import time


def tail(path, n=15):
    try:
        with open(path, 'rb') as f:
            lines = f.read().decode('utf-8', 'replace').splitlines()
        return '\n'.join(lines[-n:])
    except FileNotFoundError:
        return '(no log)'


def progress_line(path):
    """Last line that says where training is (epoch / eval / done)."""
    keys = ('Epoch: [', 'Averaged stats', 'Eval:', 'Training time', 'Reached --max_hours', '"Pr@0.5"', 'Traceback')
    try:
        with open(path, 'rb') as f:
            lines = f.read().decode('utf-8', 'replace').splitlines()
    except FileNotFoundError:
        return '(no log yet)'
    for line in reversed(lines):
        if any(k in line for k in keys):
            return line.strip()[:160]
    return (lines[-1].strip()[:160] if lines else '(empty log)')


def run_parallel(jobs, report_every=600, poll=20, strict=True):
    """jobs: list of (name, geo_aug, gpu). Returns {name: returncode}.
    strict=True: a failing job stops the others and raises (smoke tests).
    strict=False: a failing job is reported and the others keep running (long runs)."""
    os.makedirs('logs', exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    procs = {}
    for name, geo, gpu in jobs:
        log = open(os.path.join('logs', name + '.log'), 'w')
        procs[name] = (subprocess.Popen(['bash', 'scripts/run_one.sh', name, geo, str(gpu)],
                                        stdout=log, stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True), log)
        print('started %s (geo_aug=%s, GPU %s), pid %d' % (name, geo, gpu, procs[name][0].pid), flush=True)

    t0, last = time.time(), 0.0
    codes = {}
    while len(codes) < len(procs):
        time.sleep(poll)
        for name, (p, log) in procs.items():
            if name in codes:
                continue
            rc = p.poll()
            if rc is None:
                continue
            log.close()
            codes[name] = rc
            print('%s finished with exit code %d after %.2f h' % (name, rc, (time.time() - t0) / 3600), flush=True)
            if rc != 0:
                print('---- tail of logs/%s.log ----\n%s' % (name, tail('logs/%s.log' % name, 40)), flush=True)
                if not strict:
                    print('continuing with the other jobs (strict=False)', flush=True)
                    continue
                for other, (q, _) in procs.items():
                    if other not in codes and q.poll() is None:
                        os.killpg(q.pid, signal.SIGTERM)
                        print('stopped %s' % other, flush=True)
                raise RuntimeError('job %s failed (exit %d)' % (name, rc))
        if time.time() - last >= report_every:
            last = time.time()
            for name, (p, _) in procs.items():
                if name not in codes:
                    print('[%.2f h] %s: %s' % ((time.time() - t0) / 3600, name, progress_line('logs/%s.log' % name)),
                          flush=True)
    for name in procs:
        ok = os.path.exists(os.path.join('results', name, 'metrics.json'))
        print('%s: %s' % (name, 'results/%s/metrics.json written' % name if ok else 'NO METRICS'), flush=True)
    return codes


if __name__ == '__main__':
    run_parallel([tuple(j.split(':')) for j in sys.argv[1:]])
