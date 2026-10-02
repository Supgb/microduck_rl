import swanlab

swanlab.init(
    project="microduck-standing",
)

swanlab.sync_tensorboard_torch()

from mjlab.scripts.train import main

if __name__ == "__main__":
    main()
