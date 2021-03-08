import argparse
from model import StyleTransfer

def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--alpha',
        type=float,
        default=1e5
    )