import tensorflow.experimental.numpy as znp  # this is the name in zfit as well, keep it
from zfit import z
import tensorflow as tf
import functools
from tensorflow.python.util.object_identity import Reference
from collections import OrderedDict
import dill

jit = tf.function(autograph=False)

cache = OrderedDict()

class InvalidateCacheGlobal:

    def invalidate_cache(self):
        cache.clear()


class ZfitParameteter(InvalidateCacheGlobal):

    def __init__(self, x):
        self.x = x

    def get_value(self):
        return self.x

    def set_value(self, value):
        self.invalidate_cache()  # can be moved to a decorator, but not important now
        self.x = value

class FunctionArguments():
  def __init__(self, args):
    self.input_args = args
    args_list = list()
    for arg in args:
      if tf.is_tensor(arg):
        args_list.append(arg.ref())
      else:
        args_list.append(arg)
    self.args = tuple(args_list)

  def __hash__(self):
        return hash(dill.dumps(self.input_args))

  def __eq__(self, other):
        print("eq")
        return isinstance(other, FunctionArguments) and self._compare_args(other)

  def _compare_args(self, other):
    for arg1, arg2 in zip(self.args, other.args):
      if type(arg1) == Reference:
        arg1 = arg1.deref()
      if type(arg2) == Reference:
        arg2 = arg2.deref()
      if not arg1.__eq__(arg2):
        return False
    return True

  def __repr__(self):
    result = ''
    for arg in self.args:
      if type(arg) == Reference:
        result += str(arg.deref().numpy())
      else:
        result += str(arg)
      result += ' '
    return result


class ZfitFunc():
    def cache(func):
        @functools.wraps(func)
        def wraper(self, *args):
          func_args = FunctionArguments(args)
          if str(func.__name__) in cache:
            if func_args in cache[str(func.__name__)]:
              return cache[str(func.__name__)][func_args]
            else:
              cache[str(func.__name__)][func_args] = func(self, *args)
          else:
            func_dict = OrderedDict()
            func_dict[func_args] = func(self, *args)
            cache[str(func.__name__)] = func_dict
          return cache[str(func.__name__)][func_args]
        return wraper

    def __init__(self, param) -> None:
        self.param = param

    @cache
    @jit
    def func(self, data):  # cache this value and the gradient
        return data * self.param.get_value()

    @cache
    @jit
    def integral(self, lower, upper):
        # cache
        return upper * self.func(upper) - lower * self.func(lower)



param = ZfitParameteter(znp.array(22.))
func = ZfitFunc(param)
func.func(znp.array(15.))
print(cache)
func.func(znp.array(21.))
print(cache)
print(func.integral(znp.array(4.), znp.array(5.)))