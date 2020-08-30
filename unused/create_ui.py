import pandas as pd
import numpy as np
import matplotlib.image as mpimg
import os
import random
import argparse

from utils import preprocess1, preprocess2, lane_image, iIMG_H, iIMG_W


parser = argparse.ArgumentParser(description='create different types of training data')
parser.add_argument('-n', '--name', type=str, metavar='', required=True, help='name of the track')
group = parser.add_mutually_exclusive_group()
group.add_argument('-tf2', '--tensorflowgpu', action='store_true', help='create data for tf2 training')
group.add_argument('-la', '--lane', action='store_true', help='create lane detected data')
group.add_argument('-gr', '--grey', action='store_true', help='create grey scale data')
args = parser.parse_args()


# == FUNCTION ==========================================================================================================
def create_data_1(training_data, edge_detect, iteration=5, lucky_num=0.6):
    processed_training_data = []

    if iteration > 0:
        for i in range(0, iteration):
            print("Iteration: {}...".format(i))

            for frame in training_data:
                center = frame[0]
                left = frame[1]
                right = frame[2]
                angle = frame[3]

                if np.random.rand() < lucky_num:
                    image, angle = preprocess2(center, left, right, angle, edge_detect=edge_detect)
                    image = preprocess1(image, edge_detect=edge_detect)

                else:
                    image = preprocess1(center, edge_detect=edge_detect)

                processed_training_data.append([image, angle])

    else:
        print("Incorrect parameter <iteration>!")

    return processed_training_data


np.random.seed(0)


def main(edge_detect, file_name, generator):
    # == CREATE DATA ===================================================================================================
    if edge_detect:
        print("Generating edge images")

    if generator:
        print("Generating data for generator")

    print("Loading data...")
    data_dir = "./images/"
    log_name = "driving_log.csv"
    data_df = pd.read_csv(os.path.join(data_dir, log_name),
                          names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    training_df = data_df[['center', 'left', 'right', 'steering']].values
    training_data = []

    for frame in training_df:
        center = mpimg.imread(frame[0])
        left = mpimg.imread(frame[1])
        right = mpimg.imread(frame[2])
        angle = frame[3]

        if edge_detect:
            center = lane_image(center)
            left = lane_image(left)
            right = lane_image(right)

        training_data.append([center, left, right, angle])

    if generator:
        # train_v2
        cloned_data = training_data
    else:
        # train_v1
        print("Cloning data...")
        cloned_data = create_data_1(training_data, edge_detect=edge_detect, iteration=10)

    random.shuffle(cloned_data)

    # == SAVING DATA ===================================================================================================
    print("Saving data...")
    np.save(file_name, cloned_data)


if __name__ == '__main__':
    # main(edge_detect=False)
    name = args.name
    if args.tensorflowgpu:
        data_type = 'tf2'
        file_name = 'SDC_{}_{}.npy'.format(name, data_type)
        main(edge_detect=False, file_name=file_name, generator=True)
    elif args.lane:
        data_type = 'lane'
        file_name = 'SDC_{}_{}.npy'.format(name, data_type)
        main(edge_detect=True, file_name=file_name, generator=True)
    elif args.grey:
        data_type = 'grey'
        file_name = 'SDC_{}_{}.npy'.format(name, data_type)
    else:
        print('TypeError: "type" ')
        pass
