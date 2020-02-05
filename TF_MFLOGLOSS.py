import tensorflow as tf
import os, random, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 7-73行初始化各类变量，如学习率、bu、bi、Uu、Vi、T等
# ******************************************** #
au = 0.001
av = 0.001
Bu = 0.001
Bv = 0.001
learning_rate = 0.01
T = 100 # pdf要求100，这里设置为1用于测试
k = 5
re = {}

n = 943
m = 1682
p = 3
bu = [0]*(n+1)
bi = [0]*(m+1)
record_count = 0

user_set = set()
item_set = set()
whole_A = []
P = []
observed_item = set()
Iu = {}
for line in open('ml-100k/u1.base.occf', 'r', encoding='utf-8'):
	record = line.split()
	user_set.add(int(record[0]))
	item_set.add(int(record[1]))
	P.append((int(record[0]), int(record[1]), 1.0))
	bu[int(record[0])] += 1
	bi[int(record[1])] += 1
	record_count += 1
	observed_item.add(int(record[1]))
	if(int(record[0])) not in Iu.keys():
		Iu[int(record[0])] = (int(record[1]),)
	else:
		Iu[int(record[0])] += (int(record[1]),)

test_Iu = {}
for line in open('ml-100k/u1.test.occf', 'r', encoding='utf-8'):
	record = line.split()
	if(int(record[0])) not in test_Iu.keys():
		test_Iu[int(record[0])] = (int(record[1]),)
	else:
		test_Iu[int(record[0])] += (int(record[1]),)

temp_whole_A = []
u = record_count / n / m
for user in range(1, len(bu)):
	bu[user] = bu[user] / m - u
for item in range(1, len(bi)):
	bi[item] = bi[item] / n - u
for user in user_set:
	for item in item_set:
		temp_whole_A.append((user, item, 1.0))
temp_whole_A = list(set(temp_whole_A) - set(P))
for i in range(len(temp_whole_A)):
	whole_A.append((temp_whole_A[i][0], temp_whole_A[i][1], -1.0))

Uu = [0]*(m+1)
Vi = [0]*(m+1)
d = 20
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
# ******************************************** #

# 78-89行设置placeholder，以及四个variable
# ******************************************** #
user = tf.placeholder(tf.int32, name='user')
item = tf.placeholder(tf.int32, name='item')
rui = tf.placeholder(tf.float32, name='rui')

bu_embeddiing = tf.Variable(bu)
bu_user = tf.nn.embedding_lookup(bu_embeddiing, user)
bi_embedding = tf.Variable(bi)
bi_item = tf.nn.embedding_lookup(bi_embedding, item)
Uu_embedding = tf.Variable(Uu)
Uu_tensor = tf.reshape(tf.nn.embedding_lookup(Uu_embedding, user), shape=[1, d])
Vi_embedding = tf.Variable(Vi)
Vi_tensor = tf.reshape(tf.nn.embedding_lookup(Vi_embedding, item), shape=[d, 1])
# ******************************************** #
# 设置预测值_rui
_rui = bu_user + bi_item + tf.reduce_sum(tf.matmul(Uu_tensor, Vi_tensor))
# 设置loss和optimizer
loss = tf.log(1.0+tf.exp(-1.0*rui*_rui)) + 0.5*au*tf.reduce_sum(tf.square(Uu_tensor)) \
 + 0.5*av*tf.reduce_sum(tf.square(Vi_tensor)) + 0.5*Bu*tf.square(bu_user) + 0.5*Bv*tf.square(bi_item)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)   

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(tf.global_variables_initializer())

	# 根据user、item的值传入feed_dict并训练
	for i in range(T):
		A = random.sample(whole_A, p*record_count)
		union_pairs = A + random.sample(P, record_count)
		union_pairs = random.sample(union_pairs, len(union_pairs))
		# user_value_list = []
		# item_value_list = []
		# rui_value_list = []
		# for user_value, item_value, rui_value in union_pairs:
		# 	user_value_list.append(user_value)
		# 	item_value_list.append(item_value)
		# 	rui_value_list.append(rui_value)
		# input_queue = tf.train.slice_input_producer([rui_value_list, user_value_list, item_value_list], shuffle=False)
		# rui_value_batch, user_value_batch, item_value_batch = sess.run(tf.train.batch(input_queue, batch_size=60, num_threads=20, capacity=int(len(rui_value_list)/60)+2))
		# sess.run(optimizer, feed_dict={rui:rui_value_batch, user:user_value_batch, item:item_value_batch})
		for user_value, item_value, rui_value in union_pairs:
			sess.run(optimizer, feed_dict={rui:rui_value, user:user_value, item:item_value})

	# 为每个用户找到top-k个物品，用户为re的key，top-k个物品为re的value
	for user_value in test_Iu.keys():
		ranking_list = {}
		for item_value in (observed_item - set(Iu[user_value])):
			if bu[user_value] != 0 and bi[item_value] != 0:
				_rui_value = sess.run(_rui, feed_dict={user:user_value, item:item_value})
				ranking_list[item_value] = _rui_value
		top_k_pairs = sorted(ranking_list.items(), key = lambda x:x[1], reverse = True)[:k]
		re[user_value] = [pair[0] for pair in top_k_pairs]

	# 找出re前k个item中与test的交集个数，除以k，最后除以test用户数
	Pre_k = 0.0 
	for key in test_Iu.keys():
		Pre_k += (len(set(test_Iu[key]).intersection(set(re[key]))) / (k*len(test_Iu.keys())))
	print("Pre_5: "+str(Pre_k))

	# 找出re前k个item中与test的交集个数，除以用户对应的test中的item个数，最后除以test用户数
	Rec_k = 0.0
	for key in test_Iu.keys():
		Rec_k += (len(set(test_Iu[key]).intersection(set(re[key]))) / (len(test_Iu[key])*len(test_Iu.keys())))
	print("Rec_5: "+str(Rec_k))