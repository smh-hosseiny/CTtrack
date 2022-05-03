from test import Test
from train import Train
import argparse
import os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('action',  metavar="-action", choices=['train', 'test'], help='Action to take')
    parser.add_argument('data', metavar="-data", help="path to a directory containing dwi and bvals/bvecs")
    parser.add_argument("-labels", help="path to a directory containing ground truth SH.")
    parser.add_argument("-bm", help="path to brain mask", action="store")
    parser.add_argument("-wm", help="path to white matter mask", action="store")
    parser.add_argument("-trained_model_dir", metavar="model_dir", help="trained model (.ckpt)")
    parser.add_argument("-save_dir", help="directory to save trained model or generated tractogram", default=os.getcwd())
    parser.add_argument("-algorithm", action="store", default = "deterministic", help="Tractography algorithm (deterministic or probabilistic)")
    parser.add_argument("-num_tracts", action="store", default = 120000, help="number of streamlines (default 80000)", type=int)
    parser.add_argument("-min_length", action="store", default = 10, help="min length of streamlines (default 10 mm)", type=int)
    parser.add_argument("-max_length", action="store", default = 250, help="max length of streamlines (default 250 mm)", type=int)
    parser.add_argument("-train_batch_size", action="store", default = 256, help="train atch size", type=int)
    parser.add_argument("-track_batch_size", action="store", default = 1000, help="track batch size", type=int)
    parser.add_argument("-lr", "--learning_rate", default = 0.0001, help="learning rate", type=int)
    parser.add_argument("--epochs", default = 1, help="number of epochs", type=int)
    parser.add_argument("-dropout_prob", default = 0.1, help="dropout probability", type=int)
    parser.add_argument("-split_ratio", default = 0.8, help="train test split ratio", type=int)

    args = parser.parse_args()
    print(args)

    if args.action == 'train':
        trainer = Train(args)
        trainer.train()

    else:
        tracker = Test(args)
        tractogram = tracker.track()
