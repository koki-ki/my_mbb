# coding: utf-8¥
import os
import random
import sys
from timeit import default_timer as timer
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

SCRIPT_NAME = os.path.splitext(__file__)[0]
SCRIPT_ABS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_ABS_DIR, "..")))
sys.path.append('..')
sys.path.append('../lib')
sys.path.append('../../res/')

from pbc.trainer.MBBTrainer import MBBTrainer  # noqa
from pbc.model.TrainingModel import MaximumBayesBoundarynessTraining  # noqa
from pbc.model.OptimizerModel import GDOptimizer  # noqa
from pbc.model.ClassifierModel import MultiClassKmeansPrototypeClassifier  # noqa
from lib.load_data import load_data  # noqa
from lib.argument import get_MBBT_args  # noqa


def main():
    # set random seed
    seed = 0
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    tf.set_random_seed(seed)

    # Get arguments
    config = get_MBBT_args()

    if not os.path.isdir(config.output):
        os.makedirs(config.output)

    if config.name == "":
        config.name = "%s-%s" % (os.path.basename(SCRIPT_NAME),
                                 os.path.basename(config.dataset).split(".")[0])

    x_train, y_train, x_test, y_test = load_data(config.dataset,
                                                 delimiter=",",
                                                 training=config.training_size,
                                                 testing=config.testing_size,
                                                 save_split_data=True,
                                                 load_split_data=True,
                                                 normalize_by_z_score=True,
                                                 make_one_hot_label=True,
                                                 randomize=True)

    cy_train, N_cy_train = np.unique(
        np.argmax(y_train, axis=1), return_counts=True)

    minimum_training_sample = np.amin(N_cy_train)

    if config.proto > minimum_training_sample:
        print("WARNING: Too large prototypes (config.proto > min(N_cy))")
        print("Overrite config.proto as %s" % minimum_training_sample)
        config.proto = minimum_training_sample

    # Making Classifier Model
    classifier_model = MultiClassKmeansPrototypeClassifier(
        config, x_train, y_train)

    # Making Trainer Model
    trainer_model = MaximumBayesBoundarynessTraining(classifier_model)

    # Making Optimizer Model
    optimizer_model = GDOptimizer(classifier_model, trainer_model)

    # Start session
    sess = tf.Session(graph=tf.get_default_graph())

    # Training model
    trainer = MBBTrainer(classifier_model,
                         trainer_model,
                         optimizer_model,
                         sess,
                         config,
                         x_train,
                         y_train,
                         x_test,
                         y_test)

    # Training
    time = trainer.train()

    # res_path = '../../res/' + str(time) + '.txt'
    time_result = trainer.conf.output + '/' + str(time) + '.txt'
    f = open(time_result, 'w')
    f.close()

if __name__ == "__main__":
    print("START: %s" % SCRIPT_NAME)
    start = timer()
    main()
    end = timer()
    print("DONE: %s" % SCRIPT_NAME)

    
