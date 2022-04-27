import tensorflow.experimental.numpy as znp  # this is the name in zfit as well, keep it
from zfit import z
import tensorflow as tf

jit = tf.function()


class ZfitParameter:
    def __init__(self, x):
        self.x = x

    def get_value(self):
        return self.x

    def set_value(self, value):
        # self.invalidate_cache()
        self.x = value


class ZfitFunc:
    def __init__(self, param) -> None:
        self.param = param

    @jit
    def func(self, data):
        return data * self.param.get_value()

    @jit
    def integral(self, lower, upper):
        return upper * self.func(upper) - lower * self.func(lower)


def cache_value(cache: tf.Variable, flag: tf.Variable, func, *args):
    def autoset_func():
        tf.print('Function ' + func.__name__ + ' evaluated with args ' + str(args))
        val = func(*args)
        cache.assign(val, read_value=False)
        flag.assign(True, read_value=False)
        return cache

    def use_cache():
        tf.print('Cache for function ' + func.__name__ + ' with args ' + str(args) + ' was used')
        return cache

    val = tf.cond(flag, use_cache, autoset_func)
    return val


def do_value_caching(flag, value):
    flag.assign(value)


class CachedZfitFunc(ZfitFunc):
    def __init__(self, param, cache_tol=None):
        super().__init__(param)
        self.func_cached_argument = None
        self.func_cached_value = None
        self.func_cache_valid = tf.Variable(initial_value=False, trainable=False)

        self.integral_cached_argument = None
        self.integral_cached_value = None
        self.integral_cache_valid = tf.Variable(initial_value=False, trainable=False)

        self.do_func_caching = tf.Variable(initial_value=True, trainable=False)
        self.do_integral_caching = tf.Variable(initial_value=True, trainable=False)

        self.cache_tolerance = 1e-8 if cache_tol is None else cache_tol

    def invalidate_func_cache(self):
        self.do_func_caching.assign(False)
        self.invalidate_integral_cache()

    def invalidate_integral_cache(self):
        self.do_integral_caching.assign(False)

    def func(self, data):
        tf.print("Overwritten func is called")
        func_cached_argument = self.func_cached_argument
        if func_cached_argument is None:
            func_cached_argument = tf.Variable(znp.zeros(shape=tf.shape(data)), trainable=False, validate_shape=False,
                                               dtype=tf.float64)
            self.func_cached_argument = func_cached_argument

        func_cached_value = self.func_cached_value
        if func_cached_value is None:
            func_cached_value = tf.Variable(znp.zeros(shape=tf.shape(data)), trainable=False, validate_shape=False,
                                            dtype=tf.float64)
            self.func_cached_value = func_cached_value

        args_same = tf.math.reduce_all(tf.math.abs(data - self.func_cached_argument) < self.cache_tolerance)
        self.func_cache_valid.assign(tf.math.logical_and(args_same, self.do_func_caching), read_value=False)
        self.func_cached_argument.assign(data, read_value=False)
        self.do_func_caching.assign(True)
        value = cache_value(func_cached_value, self.func_cache_valid, super().func, data)
        return value

    def integral(self, lower, upper):
        tf.print("Overwritten integral is called")
        input_args = tf.stack([lower, upper])
        integral_cached_argument = self.integral_cached_argument
        if integral_cached_argument is None:
            integral_cached_argument = tf.Variable(znp.zeros(shape=tf.shape(input_args)), trainable=False,
                                                   validate_shape=False,
                                                   dtype=tf.float64)
            self.integral_cached_argument = integral_cached_argument

        integral_cached_value = self.integral_cached_value
        if integral_cached_value is None:
            integral_cached_value = tf.Variable(znp.zeros(shape=tf.shape(lower)), trainable=False, validate_shape=False,
                                                dtype=tf.float64)
            self.integral_cached_value = integral_cached_value

        args_same = tf.math.reduce_all(tf.math.abs(input_args - self.integral_cached_argument) < self.cache_tolerance)
        self.integral_cache_valid.assign(tf.math.logical_and(args_same, self.do_integral_caching), read_value=False)
        self.integral_cached_argument.assign(input_args, read_value=False)
        value = cache_value(integral_cached_value, self.integral_cache_valid, super().integral, lower, upper)
        self.do_integral_caching.assign(True)
        return value


tf.config.run_functions_eagerly(False)

param = ZfitParameter(znp.array(1.))
cached_func = CachedZfitFunc(param)

print(cached_func.func(znp.array(5.)))
print('\n')
print(cached_func.func(znp.array(5.)))
print('\n')
print(cached_func.func(znp.array(6.)))
print('\n')

print(cached_func.integral(lower=znp.array(5.), upper=znp.array(6.)))
print('\n')
print(cached_func.integral(lower=znp.array(4.), upper=znp.array(5.)))
print('\n')

param.set_value(znp.array(2.))  # doesnt change class attribute in graph mode
cached_func.invalidate_func_cache()

print(cached_func.func(znp.array(5.)))
print('\n')
print(cached_func.func(znp.array(5.)))
print('\n')

cached_func.invalidate_integral_cache()
print(cached_func.integral(znp.array(4.), znp.array(5.)))
print('\n')
print(cached_func.integral(znp.array(3.), znp.array(4.)))
