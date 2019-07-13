import argparse
import ast


def define_parser():
    parser = argparse.ArgumentParser(description='Classification by BiLSTM')
    parser.add_argument('--seq',
                        required=False,
                        metavar=False,
                        help='Use sequence model or single model.',
                        type=ast.literal_eval,
                        dest='seq')
    parser.add_argument('--bs',
                        required=False,
                        metavar=32,
                        help="Batch Size",
                        type=int,
                        dest='bs')
    parser.add_argument('--ep',
                        required=False,
                        metavar=300,
                        help="Epoch",
                        type=int,
                        dest='ep')
    args = parser.parse_args()
    return args
