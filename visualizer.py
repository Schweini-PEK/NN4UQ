import argparse
import logging

from utils import analysis
from utils import kits

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", "-s", default=False, type=bool, help="Save figure")
    parser.add_argument("--model", "-m", help="Model for predicting")
    parser.add_argument("--dataset", "-d", default='dataset/NS_truth_x3a5.pkl', help="Test set")
    parser.add_argument("--loss_path", help="Path of Ray json file")
    parser.add_argument("--model_dir", default='results/compare', help="Path of Ray json file")
    parser.add_argument("--loss_dir", help="Path of Ray json file")
    parser.add_argument("--delta", default=False, help="Time lag related data")
    args = parser.parse_args()

    models_pth = kits.get_pth_from_dir(args.model_dir)
    for pth in models_pth:
        analysis.forecast(model_path=pth, truth_path=args.dataset, delta=args.delta, save_fig=args.save)

    if args.loss_dir is not None:
        analysis.plot_loss(args.loss_dir)
