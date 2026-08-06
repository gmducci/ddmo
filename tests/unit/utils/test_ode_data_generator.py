import numpy as np
import pytest

from ddmo import ODEDataGenerator


def exponential_decay(state, _time):
    return -0.5 * np.asarray(state)


def test_ode_data_generator_builds_batch_and_data_constructor():
    time = np.linspace(0.0, 1.0, 6)
    generator = ODEDataGenerator(
        exponential_decay,
        initial_conditions=[np.array([1.0]), np.array([2.0])],
        time_array=time,
        noise_level=0.0,
    )

    batch = generator.generate()
    data = batch.to_data_constructor(include_derivatives=True)

    assert batch.x_clean.shape == (2, 6, 1)
    assert batch.x_noisy.shape == (2, 6, 1)
    assert batch.x_dot_clean.shape == (2, 6, 1)
    np.testing.assert_allclose(batch.x_clean, batch.x_noisy)
    np.testing.assert_allclose(batch.x_dot_clean, -0.5 * batch.x_clean)

    assert data.x_train.shape == (2, 6, 1)
    assert data.x_dot.shape == (2, 6, 1)
    assert data.number_initial_conditions == 2
    assert data.dimension == 1


def test_ode_data_generator_noise_is_reproducible():
    kwargs = {
        "ode": exponential_decay,
        "initial_conditions": [np.array([1.0])],
        "t_start": 0.0,
        "t_end": 1.0,
        "sampling_frequency": 5.0,
        "noise_level": 0.1,
        "seed": 123,
    }

    first = ODEDataGenerator(**kwargs).generate()
    second = ODEDataGenerator(**kwargs).generate()

    np.testing.assert_allclose(first.x_noisy, second.x_noisy)
    assert not np.allclose(first.x_clean, first.x_noisy)


def test_ode_data_generator_can_clip_noisy_states():
    def zero_dynamics(state, _time):
        return np.zeros_like(state)

    batch = ODEDataGenerator(
        zero_dynamics,
        initial_conditions=[np.array([0.0])],
        t_start=0.0,
        t_end=1.0,
        sampling_frequency=5.0,
        noise_level=1.0,
        seed=1,
        clip_min=1e-8,
    ).generate()

    assert np.all(batch.x_noisy >= 1e-8)


def test_ode_data_generator_requires_time_grid():
    with pytest.raises(ValueError, match="sampling_frequency is required"):
        ODEDataGenerator(exponential_decay, initial_conditions=[np.array([1.0])])
