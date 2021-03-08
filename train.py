import argparse
from model import StyleTransfer
from dataloader import get_train_loader
def main(args):
    train_dl, style_img = get_train_loader(4)
    model = StyleTransfer(style_img, ['relu2_2'], ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], alpha=1e3)

    x, y = next(iter(train_dl))
    print(model.train_batch(x))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--alpha',
        type=float,
        default=1e5
    )
    args = parser.parse_args()
    main(args)