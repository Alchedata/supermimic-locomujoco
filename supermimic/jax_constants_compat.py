"""Process-local compatibility for this pinned JAX's constant-hoisting option.

The installed implementation concatenates tuples with sequence-typed inputs that
can be lists. Normalize these container types without touching array values or
installed packages. Needed when passing the 2GB motion library as const operands.
"""
import inspect
from functools import wraps
from jax._src.interpreters import pxla


def install():
    original=pxla.hoist_constants_as_args
    if getattr(original,'_sequence_compat',False):return
    signature=inspect.signature(original)
    @wraps(original)
    def compatible(*args,**kwargs):
        bound=signature.bind(*args,**kwargs)
        for name in ['global_in_avals','in_shardings','in_layouts','donated_invars','inout_aliases']:
            value=bound.arguments[name]
            if value is not None:bound.arguments[name]=tuple(value)
        info=bound.arguments['all_args_info']
        debug=info.debug_info
        if debug.arg_names is not None:debug=debug._replace(arg_names=tuple(debug.arg_names))
        bound.arguments['all_args_info']=info._replace(in_avals=tuple(info.in_avals),debug_info=debug)
        return original(*bound.args,**bound.kwargs)
    compatible._sequence_compat=True
    pxla.hoist_constants_as_args=compatible
