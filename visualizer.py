import argparse
import logging

import matplotlib.pyplot as plt

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
    parser.add_argument("--loss_dir", default='results/losses1', help="Path of Ray json file")
    parser.add_argument("--delta", default=False, help="Time lag related data")
    args = parser.parse_args()

    models_pth = kits.get_pth_from_dir(args.model_dir)
    for pth in models_pth:
        analysis.forecast(model_path=pth, truth_path=args.dataset, delta=args.delta, save_fig=args.save)

    losses_pth = kits.get_pth_from_dir(args.loss_dir)
    fig, ax = plt.subplots()
    scalar_map = kits.color_wheel(len(losses_pth), theme='prism')
    for i, pth in enumerate(losses_pth):
        loss = analysis.get_loss_from_ray(pth)
        color = scalar_map.to_rgba(i)
        ax.plot(loss, color=color, label=pth.split('/')[-1].split('.')[0])
    leg = ax.legend()
    plt.xlabel('Iterations')
    plt.ylabel('L1 Loss')
    plt.title('Validation Losses Comparison')
    plt.show()
