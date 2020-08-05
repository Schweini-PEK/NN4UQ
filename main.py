import argparse

from utils import analysis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", "-s", default=False, type=bool, help="Save figure")
    parser.add_argument("--model", "-m", help="Model for predicting")
    parser.add_argument("--dataset", "-d", help="Test set")
    parser.add_argument("--loss_path", help="Path of result.json")
    args = parser.parse_args()

    analysis.forecast(model_path=args.model, truth_path=args.dataset, save_fig=args.save)
    # loss = analysis.get_loss_from_ray(args.loss_path)
