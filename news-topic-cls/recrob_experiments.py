from pytorch_lightning import Trainer, seed_everything, loggers
from argparse import ArgumentParser
from pytorch_lightning.callbacks import LearningRateLogger
import json

from models.rec_rob import RecRob

seed_everything(0)


def main(hparams):
    hparams.data_path += "x"
    experiments = {
        "0": [1],
        "1": [1],
        "2": [1],
        "3": [1],
        "4": [1]
    }

    loss_weights = {
        "1": 1.0,
        "5": 3.25,
        "21": 5.5
    }

    for seed, labels in experiments.items():
        for amount_labels in labels:
            splitted_path = hparams.data_path.split("/")
            splitted_path[-1] = str(seed)
            hparams.data_path = "/".join(splitted_path)
            hparams.amount_labels = amount_labels
            hparams.loss_weight = loss_weights[str(amount_labels)]

            model = RecRob(hparams)

            name = "recrob"
            logger = loggers.TensorBoardLogger(
                save_dir=hparams.output_dir,
                name=name,
                version="{}-{}".format(str(amount_labels),
                                       str(seed)),
                log_graph=True
            )

            lr_logger = LearningRateLogger(logging_interval="step")
            trainer = Trainer(default_root_dir=logger.log_dir + "/checkpoints/",
                              logger=logger,
                              log_save_interval=10,
                              callbacks=[lr_logger],
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_path",
                        default="../../data/train-eval-split")
    parser.add_argument("--amount_labels", type=int,
                        default=5, choices=[1, 5, 21])
    parser.add_argument("--lm_path",
                        default="../../data/models/language_models/roberta-lm")
    parser.add_argument("--output_dir",
                        default="../../data/models/classifier/")

    parser.add_argument("--rnn_hidden_size", type=int, default=384)
    parser.add_argument("--cls_hidden_size", type=int, default=192)
    parser.add_argument("--split_size", type=int, default=200)
    parser.add_argument("--shift", type=int, default=50)
    parser.add_argument("--scheduler", action="store_true")

    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate_ext", type=float, default=0.001)
    parser.add_argument("--learning_rate_lm", type=float, default=1e-5)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--gradient_clip_val", type=float, default=0.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--tpu_cores", type=int, default=None)
    parser.add_argument("--amp_level", default="O0")

    parser.add_argument("--auto_lr_find", action="store_true")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    parser.add_argument("--fast_dev_run", action="store_true")

    main(parser.parse_args())
