import numpy as np

from ddmo.utils.data_constructor import DataConstructor

class TestDataConstructor:
    def test_initialization_basic(self):
        """Test that DataConstructor calculates dimensions correctly."""
        # Simple data: 10 points, 2 variables
        x_train = np.zeros((10, 2))
        time = np.linspace(0, 1, 10)
        x_dot = np.zeros((10, 2))
        x_0 = [np.array([0, 0])]
        
        dc = DataConstructor(
            x_train=x_train,
            time_array=time,
            x_dot=x_dot,
            x_0=x_0,
            x_raw=x_train
        )
        
        assert dc.dimension == 2
        assert dc.number_initial_conditions == 1

    def test_initialization_post_init_logic(self):
        """Test that dimension is derived from x_train shape."""
        # 3 variables
        x_train = np.zeros((5, 3))
        
        dc = DataConstructor(
            x_train=x_train,
            time_array=np.zeros(5),
            x_dot=np.zeros((5, 3)),
            x_0=[np.zeros(3)] * 5, # 5 initial conditions
            x_raw=x_train
        )
        
        assert dc.dimension == 3
        assert dc.number_initial_conditions == 5
