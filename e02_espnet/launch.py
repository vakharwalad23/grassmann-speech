"""asr.sh runs every tool as "${python} -m espnet2.bin.<tool>"; training and inference pass through here to register the encoder."""

import os
import runpy
import sys

NEEDS_ENCODER = ("espnet2.bin.asr_train", "espnet2.bin.asr_inference")


def main() -> None:
    if sys.argv[1:2] == ["-m"] and sys.argv[2] in NEEDS_ENCODER:
        from encoder import register

        register()
        sys.argv = sys.argv[2:]
        runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
    else:
        os.execv(sys.executable, [sys.executable, *sys.argv[1:]])


if __name__ == "__main__":
    main()
