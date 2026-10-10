import os

import swanlab

swanlab.init(
    project=os.getenv(
        "MICRODUCK_SWANLAB_PROJECT",
        "microduck-rl",
    ),
)

swanlab.sync_tensorboard_torch()

from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
