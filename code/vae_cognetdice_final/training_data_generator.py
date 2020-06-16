# Third party libraries
import numpy as np
import matplotlib.pyplot as plt
import itertools

# My libraries
import scene_simulator
from scene_simulator import ImageSimulator as sim


SCENE_HEIGHT = scene_simulator.SCENE_HEIGHT
SCENE_WIDTH = scene_simulator.SCENE_WIDTH
PATCH_HEIGHT = scene_simulator.PATCH_HEIGHT
PATCH_WIDTH = scene_simulator.PATCH_WIDTH


def generate_train_dataset():
    all_training_images = []
    noise_mean_list = [-0.001, 0.001]
    noise_std_list = [0.001, 0.01]
    noise_amplitude_list = [0.9, 1, 1.1]
    parameter_combinations_list = list(itertools.product(noise_mean_list, noise_std_list, noise_amplitude_list))
    for count_scene, (mu, std, amplitude) in enumerate(parameter_combinations_list):
        train_scene_grid = np.zeros(shape=(SCENE_HEIGHT, SCENE_WIDTH))
        for scene_x in range(SCENE_HEIGHT):
            for scene_y in range(SCENE_WIDTH):
                pixel_val = sim.generate_scene(scene_x, scene_y, peak_offset_x=0, peak_offset_y=0, add_noise=True, noise_mu=mu, noise_std=std, noise_amplitude=amplitude)
                train_scene_grid[scene_x][scene_y] = pixel_val

        print("Train Scene matrix: ", train_scene_grid)
        plt.imshow(train_scene_grid, 'coolwarm')
        plt.show()

        for patch_x in range(SCENE_HEIGHT-PATCH_HEIGHT):
            for patch_y in range(SCENE_WIDTH-PATCH_WIDTH):
                train_patch = train_scene_grid[patch_x:patch_x+PATCH_HEIGHT, patch_y:patch_y+PATCH_WIDTH]
                # plt.imshow(train_patch, 'coolwarm')
                # plt.show()
                all_training_images.append(train_patch)

        del train_scene_grid

    print(all_training_images)
    np.save(f'./datafolder/mytrainimagearrays', all_training_images)
    print(f'{len(all_training_images)} samples have been saved')


def generate_test_dataset():
    all_test_images = []
    noise_mean_list = [0]
    noise_std_list = [0.0]
    noise_amplitude_list = [1]
    parameter_combinations_list = list(itertools.product(noise_mean_list, noise_std_list, noise_amplitude_list))
    for count_scene, (mu, std, amplitude) in enumerate(parameter_combinations_list):
        test_scene_grid = np.zeros(shape=(SCENE_HEIGHT, SCENE_WIDTH))
        for scene_x in range(SCENE_HEIGHT):
            for scene_y in range(SCENE_WIDTH):
                pixel_val = sim.generate_scene(scene_x, scene_y, peak_offset_x=0, peak_offset_y=0, add_noise=False, noise_mu=mu, noise_std=std, noise_amplitude=amplitude)
                test_scene_grid[scene_x][scene_y] = pixel_val

        print("Test Scene matrix: ", test_scene_grid)
        plt.imshow(test_scene_grid, 'coolwarm')
        plt.show()

        for patch_x in range(SCENE_HEIGHT - PATCH_HEIGHT):
            for patch_y in range(SCENE_WIDTH - PATCH_WIDTH):
                test_patch = test_scene_grid[patch_x:patch_x + PATCH_HEIGHT, patch_y:patch_y + PATCH_WIDTH]
                print(test_patch)
                plt.imshow(test_patch, 'coolwarm')
                plt.show()
                all_test_images.append(test_patch)

        del test_scene_grid

    print(all_test_images)
    np.save(f'./datafolder/mytestimagearrays', all_test_images)
    print(f'{len(all_test_images)} samples have been saved')


def generate_test_scene_and_patch(off_x, off_y):
    test_scene = np.zeros(shape=(1, 1, SCENE_HEIGHT, SCENE_WIDTH))
    test_patch = np.zeros(shape=(1, 1, PATCH_HEIGHT, PATCH_WIDTH))

    for x in range(SCENE_HEIGHT):
        for y in range(SCENE_WIDTH):
            value1 = sim.generate_scene(x, y, peak_offset_x=0, peak_offset_y=0)
            test_scene[0][0][x][y] = value1

    test_patch[0][0][0:PATCH_HEIGHT, 0:PATCH_WIDTH] = test_scene[0][0][0+off_x:PATCH_HEIGHT+off_x, 0+off_y:PATCH_WIDTH+off_y]
    return test_scene, test_patch


if __name__ == '__main__':
    generate_train_dataset()
    generate_test_dataset()

    if True:
        test_scene_img, test_patch_img = generate_test_scene_and_patch(off_x=4, off_y=4)
        plt.imshow(test_scene_img[0][0], cmap='coolwarm')
        plt.colorbar()
        plt.show()
        plt.imshow(test_patch_img[0][0], cmap='coolwarm')
        plt.colorbar()
        plt.show()
