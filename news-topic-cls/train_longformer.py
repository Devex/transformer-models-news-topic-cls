from pytorch_lightning import Trainer, seed_everything, loggers
from argparse import ArgumentParser
import json
import torch

from models.longformer import Longformer

seed_everything(0)


def main(hparams):
    model = Longformer(hparams)

    if hparams.output_dir is not None:
        if "test" in hparams.data_path:
            name = "longformer_test"
        else:
            name = "longformer"
        logger = loggers.TensorBoardLogger(
            save_dir=hparams.output_dir,
            name=name,
            version=str(hparams.amount_labels),
            log_graph=True)
    else:
        logger = True

    trainer = Trainer(default_root_dir=logger.log_dir + "/checkpoints/",
                      logger=logger,
                      log_save_interval=10,
                      gpus=hparams.gpus,
                      tpu_cores=hparams.tpu_cores,
                      fast_dev_run=hparams.fast_dev_run,
                      max_epochs=hparams.max_epochs,
                      auto_lr_find=hparams.auto_lr_find,
                      gradient_clip_val=hparams.gradient_clip_val,
                      check_val_every_n_epoch=hparams.check_val_every_n_epoch,
                      amp_level=hparams.amp_level,
                      accumulate_grad_batches=hparams.accumulate_grad_batches)

    print("Hyperparameter:")
    print("_______________")
    print(json.dumps(vars(hparams), indent=4))
    trainer.fit(model)
    test_result = trainer.test(model)
    trainer.logger.save()
    #torch.save(model, "{}/checkpoints/{}.pth".format(logger.log_dir, logger.name))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path",
                        default="../data/train-eval-split/test")
    parser.add_argument("--amount_labels", type=int,
                        default=5, choices=[1, 5, 21])
    parser.add_argument("--lm_path", default=None)
    parser.add_argument("--output_dir",
                        default="../data/models/classifier/")

    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--max_learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--tpu_cores", type=int, default=None)
    parser.add_argument("--amp_level", default="O0")

    parser.add_argument("--auto_lr_find", action="store_true")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=5)

    parser.add_argument("--fast_dev_run", action="store_true")

    main(parser.parse_args())
