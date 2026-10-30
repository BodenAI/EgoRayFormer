"""LightningCLI entry point for EgoRayFormer.

    python run.py fit --config configs/base_unrealego.yaml --config configs/egorayformer_full.yaml ...
    EGORAYFORMER_ALLOW_TEST=1 python run.py test --config <same configs> --ckpt_path work_dirs/.../epoch=N.ckpt

The test split is evaluated only for a frozen configuration: ``run.py test`` refuses to run unless
EGORAYFORMER_ALLOW_TEST=1 (set by scripts/run_baselines.sh ALLOW_TEST=1 and scripts/retest_runs.py --allow_test).
"""

import os
import sys

# make `egorayformer` importable when running from this directory without `pip install -e .`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pytorch_lightning.cli import LightningCLI  # noqa: E402

from egorayformer.data.datamodule import EgoRayFormerDataModule  # noqa: E402,F401
from egorayformer.training.lit_module import EgoRayFormerLightningModule  # noqa: E402,F401

TEST_REFUSED = 3


def refuses_test(argv) -> bool:
    sub = next((a for a in argv[1:] if a in ("fit", "validate", "test", "predict")), None)
    return sub == "test" and os.environ.get("EGORAYFORMER_ALLOW_TEST") != "1"


def main():
    if refuses_test(sys.argv):
        sys.stderr.write("run.py test refused: the test split is evaluated only for a frozen configuration (development "
                         "decisions read the validation split). Set EGORAYFORMER_ALLOW_TEST=1 to evaluate it.\n")
        sys.exit(TEST_REFUSED)
    LightningCLI(
        EgoRayFormerLightningModule,
        EgoRayFormerDataModule,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
