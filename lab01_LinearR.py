import tensorflow as tf
X=[1,2,3]
Y=[1,2,3]

W = tf.Variable(5.0)
hypothesis = X * W

cost = tf.reduce_mean(tf.square(hypothesis-Y))
# 미분 다하기
# learning_rate = 0.1
# gradient = tf.reduce_mean((W*X-Y)*X)
# descent = W - learning_rate*gradient
# update = W.assign(descent)

#GD
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    print(step, sess.run(W))
    sess.run(train)