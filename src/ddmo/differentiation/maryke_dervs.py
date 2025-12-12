import numpy as np
import pynumdiff
import pynumdiff.optimize


def get_derivative(x_train, time_array, optimization=True):
    dt = time_array[1] - time_array[0]

    num_rows = x_train.shape[0]  # Total number of rows in x_train
    twenty_percent_index = int(
        num_rows * 0.2
    )  # Calculate 20% of the array length, assuming you want to cover 20% from the start

    # Adjust the calculation to use the 20% index
    delta_max = x_train[0, :] - x_train[twenty_percent_index, :]
    rate_max_est2 = delta_max / twenty_percent_index

    rate_max_est = np.max(np.abs(np.diff(x_train)))
    max_est_list = rate_max_est2 / dt
    # max_est_list_fin= replace_nan_with_previous(max_est_list)
    print(
        'max_est', rate_max_est, max_est_list, (np.abs(max_est_list))
    )  # , max_est_list_fin)

    cutoff_frequency1 = 2 * np.abs(max_est_list[0])  # + 0.5*np.abs(max_est_list[0])
    cutoff_frequency2 = (
        2 * np.abs(max_est_list[1])
    )  # + 0.5*np.abs(max_est_list[1])#np.max(np.abs(rate_max_est2)/dt)#max_est_list_fin#rate_max_est#1.5#3# rate_max_est*10#3  # high frequency of signal in the data.
    cutoff_frequency3 = 2 * np.abs(max_est_list[2])  # + 0.5*np.abs(max_est_list[2])

    cutoff_frequency = [cutoff_frequency1, cutoff_frequency2, cutoff_frequency3]
    # cutoff_frequency = [max(f, 0.01) for f in cutoff_frequency]
    print('cutoff', cutoff_frequency)
    # estimate by (a) counting real # peaks per second in the data or (b) look at power spectra and choose cutoff.
    tvgamma_list = []
    for cutfr in cutoff_frequency:
        log_gamma = -1.6 * np.log(cutfr) - 0.71 * np.log(dt) - 5.1
        # log_gamma =(-1)*(-1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1)
        tvgamma_ = np.exp(log_gamma)
        tvgamma_list.append(tvgamma_)
        print('tvgamma', tvgamma_, dt)
    # noise_parameters = [0, 0.1]  # mean and std
    x_hat = np.zeros_like(x_train)
    dxdt_hat = np.zeros_like(x_train)

    if optimization == True:
        for i in range(len(x_train[0, :])):
            tvgamma = tvgamma_list[i]

            params, val = pynumdiff.optimize.kalman_smooth.constant_velocity(
                x_train[:, i],
                dt,
                None,
                tvgamma=tvgamma,  # hyperparameter
                dxdt_truth=None,
            )  # no ground truth data
            # options={})

            print('tvgamma', tvgamma, 'Opt', params)
            # print([noise_parameters[1]**2, 1e-0])

            x_hat[:, i], dxdt_hat[:, i] = pynumdiff.kalman_smooth.constant_velocity(
                x_train[:, i], dt, params
            )  # [noise_parameters[1]**2, 1e-0])
            # x_hat[:, i], dxdt_hat[:, i] = pynumdiff.kalman_smooth.constant_velocity(x_train[:, i], dt, params)#, options={'smooth': True} )
            # evaluate.plot(x_train, dt, x_hat, dxdt_hat, x_truth, dxdt_truth)

    elif optimization == False:
        for i in range(len(x_train[0, :])):
            x_hat[:, i], dxdt_hat[:, i] = pynumdiff.kalman_smooth.constant_velocity(
                x_train[:, i], dt, [1e-8, 1e-8]
            )

    return x_hat, dxdt_hat


def get_derivative_test(x_train, time_array, optimization=True):
    dt = time_array[1] - time_array[0]

    num_rows = x_train.shape[0]  # Total number of rows in x_train
    twenty_percent_index = int(
        num_rows * 0.2
    )  # Calculate 20% of the array length, assuming you want to cover 20% from the start

    # Adjust the calculation to use the 20% index
    delta_max = x_train[0, :] - x_train[twenty_percent_index, :]
    rate_max_est2 = delta_max / twenty_percent_index

    rate_max_est = np.max(np.abs(np.diff(x_train)))
    max_est_list = rate_max_est2 / dt
    # max_est_list_fin= replace_nan_with_previous(max_est_list)
    print(
        'max_est', rate_max_est, max_est_list, (np.abs(max_est_list))
    )  # , max_est_list_fin)

    cutoff_frequency1 = 2 * np.abs(max_est_list[0])  # + 0.8*np.abs(max_est_list[0])
    cutoff_frequency2 = (
        2 * np.abs(max_est_list[1])
    )  # + 0.8*np.abs(max_est_list[1])#np.max(np.abs(rate_max_est2)/dt)#max_est_list_fin#rate_max_est#1.5#3# rate_max_est*10#3  # high frequency of signal in the data.
    cutoff_frequency3 = 2 * np.abs(max_est_list[2])  # + 0.8*np.abs(max_est_list[2])

    cutoff_frequency = [cutoff_frequency1, cutoff_frequency2, cutoff_frequency3]
    # cutoff_frequency = [max(f, 0.01) for f in cutoff_frequency]
    print('cutoff', cutoff_frequency)
    # estimate by (a) counting real # peaks per second in the data or (b) look at power spectra and choose cutoff.
    tvgamma_list = []
    for cutfr in cutoff_frequency:
        log_gamma = -1.6 * np.log(cutfr) - 0.71 * np.log(dt) - 5.1
        # log_gamma =(-1)*(-1.6 * np.log(cutoff_frequency) - 0.71 * np.log(dt) - 5.1)
        tvgamma_ = np.exp(log_gamma)
        tvgamma_list.append(tvgamma_)
        print('tvgamma', tvgamma_, dt)
    # noise_parameters = [0, 0.1]  # mean and std
    x_hat = np.zeros_like(x_train)
    dxdt_hat = np.zeros_like(x_train)

    if optimization == True:
        for i in range(len(x_train[0, :])):
            tvgamma = tvgamma_list[i]

            params, val = pynumdiff.optimize.kalman_smooth.constant_velocity(
                x_train[:, i],
                dt,
                None,
                tvgamma=tvgamma,  # hyperparameter
                dxdt_truth=None,
            )  # no ground truth data
            # options={})

            print('tvgamma', tvgamma, 'Opt', params)
            # print([noise_parameters[1]**2, 1e-0])

            x_hat[:, i], dxdt_hat[:, i] = pynumdiff.kalman_smooth.constant_velocity(
                x_train[:, i], dt, params
            )  # [noise_parameters[1]**2, 1e-0])
            # x_hat[:, i], dxdt_hat[:, i] = pynumdiff.kalman_smooth.constant_velocity(x_train[:, i], dt, params)#, options={'smooth': True} )
            # evaluate.plot(x_train, dt, x_hat, dxdt_hat, x_truth, dxdt_truth)

    elif optimization == False:
        for i in range(len(x_train[0, :])):
            x_hat[:, i], dxdt_hat[:, i] = pynumdiff.kalman_smooth.constant_velocity(
                x_train[:, i], dt, [1e-8, 1e-8]
            )

    return x_hat, dxdt_hat
