"""Run with: modal run modal_mixed_precision.py"""

import modal

from modal_common import image, run_script

app = modal.App("cs336-mixed-precision")


@app.function(image=image, timeout=300, scaledown_window=2, max_containers=1)
def run():
    run_script("cs336_systems/mixed-precision.py")


@app.local_entrypoint()
def main():
    run.remote()
