# coding: utf-8
import argparse
# from sklearn.utils.bunch import Bunch

import bunch


def get_MBBT_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--name", help="",
                        type=str, default="")
    parser.add_argument("-d", "--dataset", help="",
                        type=str, required=True)
    parser.add_argument("--training-size", help="Training sample size (# or ratio; default=0.5)",
                        type=float, default=0.5)
    parser.add_argument("--testing-size", help="Testing sample size (# or ratio; default=0.5)",
                        type=float, default=0.5)
    parser.add_argument("-o", "--output", help="",
                        type=str, required=True)
    parser.add_argument("--plotting-interval", help="The interval of plotting results",
                        type=int, default=1000)
    parser.add_argument("--epoch", help="# of epoch",
                        type=int, default=1000)
    parser.add_argument("--reest-B-interval", help="Re-estimation interval of B on ET",
                        type=int, default=1000)
    parser.add_argument("--reest-P-interval", help="Re-estimation interval of P on ET",
                        type=int, default=1000)
    parser.add_argument("--proto", help="# of prototype size",
                        type=int, default=50)
    parser.add_argument("--learning-rate", help="learning rate epsilon",
                        type=float, default=1.0)
    parser.add_argument("--knn-size", help="",
                        type=int, default=40)
    parser.add_argument("--sigma", help="",
                        type=float, default=0.1)                        

    args = parser.parse_args()

    return bunch.Bunch(args.__dict__)
