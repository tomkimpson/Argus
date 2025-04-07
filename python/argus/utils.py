import jax

from typing import Dict, Any, NamedTuple
from argus.types import Array


def validate_data_type(field_name: str, value: Any, datatype=jax.Array) -> None:
        if not isinstance(value, datatype):
            #todo: try to cast other types (e.g. numpy.ndarray or list to jax.Array)
            raise TypeError(f"Field '{field_name}' must be an array, got {type(value).__name__}")
        
def validate_data_shape(field_name: str, value: Any, expected_shape: tuple) -> None:    
        if value.shape != expected_shape:
            raise ValueError(f"Field '{field_name}' has wrong shape. Expected {expected_shape}, got {value.shape}")
        

def construct_fields(block_meta: Dict[str, int]) -> Dict[str, tuple]:
    x = {}
    cov = {}

    # construct state fields
    for name, value in block_meta.items():
        x[name] = (value,)
    
    # construct covariance fields
    for i, (name1, value1) in enumerate(block_meta.items()):
        for j, (name2, value2) in enumerate(block_meta.items()):
            if i == j:
                cov[name1] = (value1, value1)
            if i < j:
                cov_term = f"{name1}_{name2}"
                cov[cov_term] = (value1, value2)

    return {"x": x, "cov": cov}

def construct_namedtuple_class(classname, block_fields: Dict[str, tuple], datatype=Array) -> NamedTuple:
    fields = [(name, datatype) for name in block_fields.keys()]
    cls = NamedTuple(classname, fields) 
    setattr(cls, "_block_shapes", block_fields)

    # create with data validation
    def create(**kwargs):
        validated_data = {}
        for (name, shape) in cls._block_shapes.items():
            val = kwargs.get(name)
            if val is None:
                raise ValueError(f"Missing field: {name}")
            # validation
            validate_data_type(name, val)
            validate_data_shape(name, val, shape)
            validated_data[name] = val

        #todo: add warning if kwargs is not one of the pre-defined fields
        return cls(**validated_data)
    
    setattr(cls, "prior", staticmethod(create))
    return cls





   





        
