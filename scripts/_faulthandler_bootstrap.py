import faulthandler
import runpy
import signal
import sys

faulthandler.enable(all_threads=True)
for sig in (signal.SIGTRAP, signal.SIGUSR1, signal.SIGUSR2):
    try:
        faulthandler.register(sig, all_threads=True, chain=False)
    except (RuntimeError, ValueError):
        pass

target = sys.argv.pop(1)
runpy.run_path(target, run_name="__main__")
