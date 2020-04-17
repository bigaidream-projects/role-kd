from pba.kd import kd_model
from pba.setup import create_hparams
from pba.setup import create_parser

if __name__ == '__main__':
    FLAGS = create_parser("train")  # pylint: disable=invalid-name
    hparams = create_hparams("train", FLAGS)

    trainer = kd_model.KDModelTrainer(hparams)
    train_acc, val_acc = trainer.run_model(1)
    print train_acc, val_acc