import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

DATA_DIR = 'data'
SBM_DIR = 'submissions'
l1_nrows = 1512
l2_nrows = 1248
n_horizons = 4
y_height = 3001

if __name__ == '__main__':
    # Загружаем значения высот в узлах сетки для срезов L1 и L2
    all_data_l1 = np.load(os.path.join(DATA_DIR, 'all_data_L1.npy'))
    all_data_l2 = np.load(os.path.join(DATA_DIR, 'all_data_L2.npy'))

    assert all_data_l1.shape == (l1_nrows, y_height), 'Неправильный размер all_data_L1.npy'
    assert all_data_l2.shape == (l2_nrows, y_height), 'Неправильный размер all_data_L2.npy'

    # Загружаем горизонты
    l1_horizons_train = pd.read_csv(os.path.join(DATA_DIR, 'L1_horizons_train.csv'))
    l2_horizons_train = pd.read_csv(os.path.join(DATA_DIR, 'L2_horizons_train.csv'))

    assert l1_horizons_train.shape == (l1_nrows, n_horizons + 1)
    assert l2_horizons_train.shape == (l2_nrows, n_horizons + 1)

    # Преобразование высот для удобства отрисовки
    subset_l1 = all_data_l1[:, 500: 1100]
    subset_l1 = (subset_l1 - np.nanmin(subset_l1)) / (np.nanmax(subset_l1) - np.nanmin(subset_l1))
    subset_l2 = all_data_l2[:, 500: 1100]
    subset_l2 = (subset_l2 - np.nanmin(subset_l2)) / (np.nanmax(subset_l2) - np.nanmin(subset_l2))
    # Преобразование горизонтов для удобства отрисовки
    l1_x = l1_horizons_train['x'].values
    l1_hor_1 = l1_horizons_train['hor_1']
    l1_hor_2 = l1_horizons_train['hor_2']
    l1_hor_3 = l1_horizons_train['hor_3']
    l1_hor_4 = l1_horizons_train['hor_4']
    l2_x = l2_horizons_train['x'].values
    l2_hor_1 = l2_horizons_train['hor_1']
    l2_hor_2 = l2_horizons_train['hor_2']
    l2_hor_3 = l2_horizons_train['hor_3']
    l2_hor_4 = l2_horizons_train['hor_4']
    # Маски
    step = 1
    subset_l1_mask = np.zeros(subset_l1.shape)
    subset_l2_mask = np.zeros(subset_l2.shape)
    for i in range(5, 0, -step):
        subset_l1_mask[np.logical_and(subset_l1 < step * 0.2 * i, subset_l1 > step * 0.2 * (i - 1))] = 0.2 * i
        subset_l2_mask[np.logical_and(subset_l2 < step * 0.2 * i, subset_l2 > step * 0.2 * (i - 1))] = 0.2 * i
    # Вывод всех данных
    plt.figure()

    plt.subplot(3, 2, 1)
    plt.imshow(subset_l1.T)
    plt.colorbar()
    plt.plot(l1_x, l1_hor_1 - 500, 'r')
    plt.plot(l1_x, l1_hor_2 - 500, 'k')
    plt.plot(l1_x, l1_hor_3 - 500, 'b')
    plt.plot(l1_x, l1_hor_4 - 500, 'w')

    plt.subplot(3, 2, 2)
    plt.imshow(subset_l2.T)
    plt.colorbar()
    plt.plot(l2_x, l2_hor_1 - 500, 'r')
    plt.plot(l2_x, l2_hor_2 - 500, 'k')
    plt.plot(l2_x, l2_hor_3 - 500, 'b')
    plt.plot(l2_x, l2_hor_4 - 500, 'w')

    plt.subplot(3, 2, 3)
    plt.imshow(subset_l1_mask.T)
    plt.colorbar()
    plt.plot(l1_x, l1_hor_1 - 500, 'r')
    plt.plot(l1_x, l1_hor_2 - 500, 'k')
    plt.plot(l1_x, l1_hor_3 - 500, 'b')
    plt.plot(l1_x, l1_hor_4 - 500, 'w')

    plt.subplot(3, 2, 4)
    plt.imshow(subset_l2_mask.T)
    plt.colorbar()
    plt.plot(l2_x, l2_hor_1 - 500, 'r')
    plt.plot(l2_x, l2_hor_2 - 500, 'k')
    plt.plot(l2_x, l2_hor_3 - 500, 'b')
    plt.plot(l2_x, l2_hor_4 - 500, 'w')

    plt.subplot(3, 2, 5)
    plt.imshow((subset_l1_mask > 0.7).T)
    plt.colorbar()
    plt.plot(l1_x, l1_hor_1 - 500, 'r')
    plt.plot(l1_x, l1_hor_2 - 500, 'k')
    plt.plot(l1_x, l1_hor_3 - 500, 'b')
    plt.plot(l1_x, l1_hor_4 - 500, 'w')

    plt.subplot(3, 2, 6)
    plt.imshow((subset_l2_mask > 0.7).T)
    plt.colorbar()
    plt.plot(l2_x, l2_hor_1 - 500, 'r')
    plt.plot(l2_x, l2_hor_2 - 500, 'k')
    plt.plot(l2_x, l2_hor_3 - 500, 'b')
    plt.plot(l2_x, l2_hor_4 - 500, 'w')

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.1)
    plt.show()
