import tensorflow as tf

# x1_data = [[73., 93., 89., 96., 73.]
# x2_data = [80., 88., 91., 98., 66.]
# x3_data = [75., 93., 90., 100., 70.]]
# y_data = [152., 185., 180., 196., 142.]

# x1 = tf.placeholder(tf.float32)
# x2 = tf.placeholder(tf.float32)
# x3 = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)

# w1 = tf.Variable(tf.random_normal([1]), name='weight1')
# w2 = tf.Variable(tf.random_normal([1]), name='weight2')
# w3 = tf.Variable(tf.random_normal([1]), name='weight3')
# b = tf.Variable(tf.random_normal([1]), name='bias')
# hypothesis = x1*w1 + x2*w2 + x3*w3 + b
#코드가 길어지면 위의 방법들은 스파게티 코드가 되어 노답임 그때 매트릭스 코드 사용 필요
############

x_data=[[73., 80., 75.], [93., 88., 93.],
        [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data=[[152.], [185.], [180.], [196.], [142.]]

X=tf.placeholder(tf.float32, shape=[None, 3])
Y=tf.placeholder(tf.float32, shape=[None, 1])
W=tf.Variable(tf.random_normal([3,1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

hypothesis=tf.matmul(X, W) + b

cost=tf.reduce_mean(tf.square(hypothesis-Y))
#############

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, __ = sess.run([cost, hypothesis, train],
                            feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "cost:", cost_val, "\nPrediction:\n", hy_val)
