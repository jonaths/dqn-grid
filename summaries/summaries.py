import tensorflow as tf

def variable_summaries(var, scope_name='var'):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: la variable tf a la que se va a attach el summary
    :param scope_name: un nombre del scope
    :return:
    """
    with tf.name_scope(scope_name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def simple_summaries(var, scope_name='var'):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    :param var: la variable tf a la que se va a attach el summary
    :param scope_name: un nombre del scope
    :return:
    """
    with tf.name_scope(scope_name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        tf.summary.histogram('histogram', var)