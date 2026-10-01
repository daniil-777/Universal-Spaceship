# vlm/tools/detach.py NAME 'COMMANDS' [CAFFEINATE_FLAGS]: bash -c COMMANDS detached (new session, caffeinate -i, or the given
# flags, e.g. -dimsu for the overnight capture, ruling V1-7) from the repo root; log /Volumes/LaCie/astro-pilot/vlm/logs/NAME.log.
# Poll the log; never block a shell on the run.
import os, subprocess, sys
log = open(f'/Volumes/LaCie/astro-pilot/vlm/logs/{sys.argv[1]}.log', 'a')
flags = sys.argv[3] if len(sys.argv) > 3 else '-i'
p = subprocess.Popen(['caffeinate', flags, 'bash', '-c', sys.argv[2]], stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), env=os.environ.copy())
print('pid', p.pid, 'log', log.name)
