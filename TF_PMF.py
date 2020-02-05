import tensorflow as tf
import os, random, time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

d = 20 # number of latent dimensions
T = 100 # iteration number
n = 943 # user number
m = 1682 # item number
learning_rate = 0.01
au = av = Bu = Bv = 0.01 # tradeoff parameters
Uu = [0]*(m+1)
Vi = [0]*(m+1)
Uu[0] = [0.0]*d
Vi[0] = [0.0]*d
for i in range(1, m+1):
	Vik = []
	Uuk = []
	for x in range(d):
		Vik.append((random.random()-0.5)*0.01)
		Uuk.append((random.random()-0.5)*0.01)
	Vi[i] = Vik
	Uu[i] = Uuk
p = 0
record_list = []
for line in open('ml-100k/u1.base', 'r', encoding='utf-8'):
	record = line.split()
	record_list.append([int(record[0]), int(record[1]), float(record[2])])
	p += 1

rui = tf.placeholder(tf.float32, name='rui')
user = tf.placeholder(tf.int32, name='user')
item = tf.placeholder(tf.int32, name='item')
learning_rate_setter = tf.placeholder(tf.float32, name='learning_rate_setter')
Uu_embedding = tf.Variable(Uu)
Uu_tensor = tf.reshape(tf.nn.embedding_lookup(Uu_embedding, user), shape=[1, d])
Vi_embedding = tf.Variable(Vi)
Vi_tensor = tf.reshape(tf.nn.embedding_lookup(Vi_embedding, item), shape=[d, 1])
loss = tf.multiply(0.5, tf.square(tf.subtract(rui, tf.reduce_sum(tf.matmul(Uu_tensor, Vi_tensor))))) + \
 tf.multiply(0.5*au, tf.reduce_sum(tf.square(Uu_tensor))) + tf.multiply(0.5*av, tf.reduce_sum(tf.square(Vi_tensor)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_setter).minimize(loss)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(tf.global_variables_initializer())

	for t in range(T):
		record_list = random.sample(record_list, p)
		for record in record_list:
			sess.run(optimizer, feed_dict={learning_rate_setter:learning_rate, rui:record[2], user:record[0], item:record[1]})
		learning_rate *= 0.9

		if (t+1)%5 == 0:
			MAE = RMSE = 0.0
			result_rui_value = tf.reduce_sum(tf.matmul(Uu_tensor, Vi_tensor))
			for line in open('ml-100k/u1.test', 'r', encoding='utf-8'):
				record = line.split()
				rui_value = sess.run(result_rui_value,  feed_dict={user:int(record[0]), item:int(record[1])})
				MAE += abs(float(record[2]) - rui_value)/20000
				RMSE += ((float(record[2]) - rui_value)**2)/20000
			RMSE = RMSE**0.5
			print("第"+str(t+1)+"次，RMSE："+str(RMSE)+"  MAE："+str(MAE))
			
			# SGD_FOR_PMF： RMSE：1.3203737204328794  MAE：1.0181837253483828