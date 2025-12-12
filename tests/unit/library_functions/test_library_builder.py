import pytest
import numpy as np
from ddmo.library_functions.composer import LibraryBuilder

class TestLibraryBuilder:
    def test_fluent_chaining(self):
        """Test that methods return self and functions are added."""
        lb = LibraryBuilder()
        lb.add_polynomial(max_degree=2)\
          .add_hill("g")\
          .add_exponential("k")
        
        # Factories should be registered
        # 1 polynomial, 1 hill, 1 exponential = 3 factories
        assert len(lb._factories) == 3

    def test_build_integration(self):
        """Test building the final library list from params."""
        lb = LibraryBuilder()\
            .add_identity()\
            .add_hill("g")
            
        # Params required by HillFactory
        params = {"g": 1.0}
        
        funcs = lb.build(params)
        
        # Identity -> 1 func (x)
        # Hill -> 1 func
        assert len(funcs) == 2
        
        # Check generic execution
        # Identity
        assert funcs[0][0](5) == 5
        # Hill 1/(1+x^1) at x=1 -> 0.5
        assert funcs[1][0](1) == 0.5

    def test_add_function_static(self):
        """Test adding a simple static lambda."""
        lb = LibraryBuilder()
        lb.add_function(lambda x: x*2, "double")
        
        funcs = lb.build({})
        assert len(funcs) == 1
        f, n = funcs[0]
        assert f(10) == 20
        assert n("z") == "double(z)"

    def test_add_custom_parameterized(self):
        """Test adding a custom function that uses optimization params."""
        lb = LibraryBuilder()
        
        # f(x) = x + p
        lb.add_custom(
            func=lambda x, p: x + p["val"],
            name=lambda x, p: f"{x}+{p['val']}"
        )
        
        funcs = lb.build({"val": 10})
        assert len(funcs) == 1
        f, n = funcs[0]
        
        assert f(5) == 15
        assert n("y") == "y+10"
