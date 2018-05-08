import tensorflow as tf
import numpy as np

inputSize = 4
dataVolume = 8
hiddenNode = 1
outputSize = 1

eta = 0.1
gamma = 0.0001

epsilon_1 = 0.0000001
epsilon_2 = 0.000001
epsilon_3 = 0.000001

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def add_hiddenLayer(x, w, b):
    layer = tf.tanh(tf.matmul(x, w) + b)
    return layer

def add_outputLayer(x, w, b):
    layer = tf.matmul(x, w) + b
    return layer

x_data = np.loadtxt('input.txt')
y_data = np.loadtxt('y_data.txt')

c1_idx = np.where(y_data==1)[0]
c2_idx = np.where(y_data==-1)[0]

xs = tf.placeholder(tf.float32, [None, inputSize])
yc = tf.placeholder(tf.float32)

input_z = tf.placeholder(tf.float32, [dataVolume, outputSize])
input_z_past = tf.placeholder(tf.float32, [dataVolume, outputSize])

input_gamma = tf.placeholder(tf.float32)
input_eta = tf.placeholder(tf.float32)

# input->hidden
w_ih = weight_variable([inputSize, hiddenNode])
b_ih = bias_variable([hiddenNode])
hiddenLayer = add_hiddenLayer(xs, w_ih, b_ih)

# hidden->output
w_ho = weight_variable([hiddenNode, outputSize])
b_ho = bias_variable([outputSize])
outputLayer = add_outputLayer(hiddenLayer, w_ho, b_ho)

input_z = outputLayer
input_z_past = outputLayer

z = tf.subtract(yc , input_z)

ec = tf.reduce_sum(tf.abs(tf.reduce_sum(z, reduction_indices=[1])), reduction_indices=[0])

error_z = tf.reduce_mean(tf.reduce_sum(tf.square(z), reduction_indices=[1]))
grad_z = tf.convert_to_tensor(tf.gradients(error_z, input_z))

EuclideanDistance_grad_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z)))

delta_z = tf.subtract(input_z, input_z_past)
z_pron = tf.add(tf.subtract(input_z, tf.multiply(input_eta, grad_z)[0]), tf.multiply(input_gamma, delta_z))   # Z_pron = 𝒁 − 𝜂*𝛻𝑤𝐸(𝒁) + γ*ΔZ

error_z_pron = tf.reduce_mean(tf.reduce_sum(tf.square(tf.subtract(yc, z_pron)), reduction_indices=[1]))


def get_tau(_min, _max, _wpo):
    _tau = 1
    find_min = 10

    find_min = _min - _max

    _wpo = np.squeeze(_wpo)

    condition_1_number = _wpo * (1.0 - np.tanh(np.power(2, _tau - 1)))
    condition_2_number = np.tanh(np.power(2, _tau - 1))

    while not condition_1_number < find_min and condition_2_number > 0.5:
        _tau += 1
        condition_1_number = _wpo * (1.0 - np.tanh(np.power(2, _tau - 1)))
        condition_2_number = np.tanh(np.power(2, _tau - 1))

    print('  tau:' + str(_tau))

    return _tau

loss =  tf.reduce_mean(-tf.reduce_sum(tf.square(yc-outputLayer), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

condition_L = True

min_class1 = 10000
max_class2 = -10000

result_c1 = []
result_c2 = []

for i in range(dataVolume):
    output = sess.run(outputLayer, feed_dict={xs: np.expand_dims(x_data[i], 0)})

    condition_L = True

    if i in c1_idx:
        # print("Data [" + str(i+1) + "] is Class 1" + " result = " + str(output))
        if len(result_c1) == 0 and len(result_c2) == 0:
            min_class1 = output
            result_c1.append(output)
            continue
        elif output < max_class2:
            condition_L = False
        else:
            result_c1.append(output)
            min_class1 = min(result_c1)
            continue

    elif i in c2_idx:
        # print("Data [" + str(i+1) + "] is Class 2" + " result = " + str(output))
        if len(result_c1) == 0 and len(result_c2) == 0:
            max_class2 = output
            result_c2.append(output)
            continue
        elif output > min_class1:
            condition_L = False
        else:
            result_c2.append(output)
            max_class2 = max(result_c2)
            continue

    last_min = min_class1
    last_max = max_class2

    if condition_L == False:
        if condition_L == False:
            print('iter ' + str(i))
            print('  cramming')

            hiddenNode += 1

            if i in c1_idx:
                wpo = tf.cast(last_max - output, tf.float32)
            elif i in c2_idx:
                wpo = tf.cast(output - last_min, tf.float32)
            w_ho_val = w_ho
            w_ho_val = tf.concat([w_ho_val, wpo], 0)
            w_ho_val = sess.run(w_ho_val)
            sess.run(tf.assign(w_ho, w_ho_val, validate_shape=False))

            tau = get_tau(last_min, last_max, sess.run(wpo))
            tau_activate = tf.cast(tf.pow(2, tau - 1), tf.float32)

            temp = tf.multiply([tf.cast(1 - inputSize, tf.float32)], tau_activate)
            b_ih_val = tf.concat([b_ih, temp], 0)
            b_ih_val = tf.reshape(b_ih_val, [-1])
            sess.run(tf.assign(b_ih, b_ih_val, validate_shape=False))

            temp = tf.reshape(tf.multiply(xs, tau_activate), [inputSize, 1])
            w_ih_val = sess.run(tf.concat([w_ih, temp], 1), feed_dict={xs: np.expand_dims(x_data[i], 0)})
            sess.run(tf.assign(w_ih, w_ih_val, validate_shape=False))

            result_c1.clear()
            result_c2.clear()
            result = sess.run(outputLayer, feed_dict={xs: x_data[:i + 1]})

            for j in range(i + 1):
                if j in c1_idx:
                    result_c1.append(result[j])
                elif j in c2_idx:
                    result_c2.append(result[j])

            if (len(result_c1) == 0):
                min_class1 = 10000
            else:
                min_class1 = min(result_c1)
            if (len(result_c2) == 0):
                max_class2 = -10000
            else:
                max_class2 = max(result_c2)

            if min_class1 > max_class2:
                condition_L = True

    last_min = min_class1
    last_max = max_class2

    condition_L = False

    pruning_finish = False
    idx_hidden = 0
    ct_pruning = 0

    while pruning_finish == False:
        if hiddenNode == 1:
            pruning_finish = True
        elif idx_hidden == hiddenNode:
            pruning_finish = True
        else:
            tmp_w_ih = sess.run(w_ih)
            tmp_b_ih = sess.run(b_ih)
            tmp_w_ho = sess.run(w_ho)

            origin_w_ih = sess.run(w_ih)
            origin_b_ih = sess.run(b_ih)
            origin_w_ho = sess.run(w_ho)

            tmp_w_ih = np.ndarray.tolist(tmp_w_ih)

            tmp_w_ih = sess.run(tf.stack(tmp_w_ih[:], axis=1))
            tmp_w_ih = np.delete(tmp_w_ih, idx_hidden, axis=0)
            tmp_w_ih = sess.run(tf.transpose(tmp_w_ih, [1, 0]))

            tmp_b_ih = np.delete(tmp_b_ih, idx_hidden, axis=0)

            tmp_w_ho = np.delete(tmp_w_ho, idx_hidden, axis=0)

            sess.run(tf.assign(w_ih, tmp_w_ih, validate_shape=False))
            sess.run(tf.assign(b_ih, tmp_b_ih, validate_shape=False))
            sess.run(tf.assign(w_ho, tmp_w_ho, validate_shape=False))

            result_c1.clear()
            result_c2.clear()

            z_past = sess.run(input_z_past, feed_dict={xs: x_data[:i + 1],
                                                       yc: y_data[:i + 1],
                                                       input_eta: eta,
                                                       input_gamma: gamma})

            z_now = sess.run(input_z_past, feed_dict={xs: x_data[:i + 1],
                                                      yc: y_data[:i + 1],
                                                      input_eta: eta,
                                                      input_gamma: gamma})

            _ec = sess.run(ec, feed_dict={xs: x_data[:i + 1],
                                          yc: y_data[:i + 1],
                                          input_z: z_now,
                                          input_eta: eta,
                                          input_gamma: gamma})
            print(_ec)

            _ED_grad_z = sess.run(EuclideanDistance_grad_z, feed_dict={xs: x_data[:i + 1],
                                                                       yc: y_data[:i + 1],
                                                                       input_z: z_now,
                                                                       input_z_past: z_past,
                                                                       input_eta: eta,
                                                                       input_gamma: gamma})

            error_z_past = sess.run(error_z, feed_dict={xs: x_data[:i + 1],
                                                        yc: y_data[:i + 1],
                                                        input_z: z_now,
                                                        input_z_past: z_past,
                                                        input_eta: eta,
                                                        input_gamma: gamma})

            while (1):
                if _ec <= epsilon_1:
                    break
                else:
                    if _ED_grad_z <= epsilon_2:
                        break
                    else:
                        _ED_grad_z = sess.run(EuclideanDistance_grad_z, feed_dict={xs: x_data[:i + 1],
                                                                                   yc: y_data[:i + 1],
                                                                                   input_z: z_now,
                                                                                   input_z_past: z_past,
                                                                                   input_eta: eta,
                                                                                   input_gamma: gamma})

                        z_now = sess.run(z_pron, feed_dict={xs: x_data[:i + 1],
                                                            yc: y_data[:i + 1],
                                                            input_z: z_now,
                                                            input_z_past: z_past,
                                                            input_eta: eta,
                                                            input_gamma: gamma})

                        error_z_now = sess.run(error_z_pron, feed_dict={xs: x_data[:i + 1],
                                                                        yc: y_data[:i + 1],
                                                                        input_z: z_now,
                                                                        input_z_past: z_past,
                                                                        input_eta: eta,
                                                                        input_gamma: gamma})
                        eta_gamma_flag = False
                        ct = 0
                        while (eta_gamma_flag == False and ct < 20):
                            if error_z_now <= error_z_past:
                                eta *= 1.2
                                gamma *= 1.2
                                z_past = z_now
                                error_z_past = error_z_now
                                eta_gamma_flag = True
                                ct = ct + 1
                                break
                            else:
                                if eta > epsilon_3:
                                    eta *= 0.7
                                    gamma *= 0.7
                                    z_past_tmp = z_past
                                    z_past = z_now
                                    ct = ct + 1

                                    z_now = sess.run(z_pron, feed_dict={xs: x_data[:i + 1],
                                                                        yc: y_data[:i + 1],
                                                                        input_z: z_past,
                                                                        input_z_past: z_past_tmp,
                                                                        input_eta: eta,
                                                                        input_gamma: gamma})

                                    error_z_past = error_z_now
                                    error_z_now = sess.run(error_z, feed_dict={xs: x_data[:i + 1],
                                                                               yc: y_data[:i + 1],
                                                                               input_z: z_now,
                                                                               input_z_past: z_past_tmp,
                                                                               input_eta: eta,
                                                                               input_gamma: gamma})

                                else:
                                    break

                        _ec = sess.run(ec, feed_dict={xs: x_data[:i + 1],
                                                      yc: y_data[:i + 1],
                                                      input_z: z_now,
                                                      input_eta: eta,
                                                      input_gamma: gamma})
                        print(_ec)

            result = z_now

            for j in range(i + 1):
                if j in c1_idx:
                    result_c1.append(result[j])
                elif j in c2_idx:
                    result_c2.append(result[j])

            if (len(result_c1) == 0):
                min_class1 = 10000
            else:
                min_class1 = min(result_c1)
            if (len(result_c2) == 0):
                max_class2 = -10000
            else:
                max_class2 = max(result_c2)

            if min_class1 > max_class2:
                condition_L = True
                hiddenNode -= 1
                idx_hidden += 1
                ct_pruning += 1

            else:
                idx_hidden += 1
                condition_L = False

                sess.run(tf.assign(w_ih, origin_w_ih, validate_shape=False))
                sess.run(tf.assign(b_ih, origin_b_ih, validate_shape=False))
                sess.run(tf.assign(w_ho, origin_w_ho, validate_shape=False))

    print('  numbers of pruning node:' + str(ct_pruning))
    last_min = min_class1
    last_max = max_class2

print('-----finish!-----')
print('hidden node:' + str(hiddenNode))
print('output:')
print(result)
print('weight between input layer and hidden layer:')
print(sess.run(w_ih))
print('threshold of hidden layer:')
print(sess.run(b_ih))
print('weight between hidden layer and output node:')
print(sess.run(w_ho))
print('threshold of output node:')
print(sess.run(b_ho))

sess.close()