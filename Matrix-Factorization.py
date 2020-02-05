from numpy import mat
from numpy import linalg as la
import numpy as np
import os
import random
import datetime

def SGD_FOR_PMF():
	# part1： calculate u(global average rating value), bu(bias of user), bi(bias of item)
	# u = 3.5283058961762976, bu、bi are dictionaries, key=(user_id or item_id), value = bias of (user or item)
	count = bu_count = bi_count = 1
	u = 0.0
	bu = {}
	bi = {}
	for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base","r",encoding="utf-8"):
		# line data: user id | item id | rating | timestamp
		record = line.split()
		count += 1
	u = u / count

	# part2: initialize the parameters
	au = av = Bu = Bv = 0.01 # tradeoff parameters
	r = 0.01 # learning rate
	d = 20 # number of latent dimensions
	T = 3 # iteration number
	n = 943 # user number
	m = 1682 # item number
	V = (np.random.random((m,d)) - 0.5)*0.01
	U = (np.random.random((n,d)) - 0.5)*0.01 # initialize matrixs
	for t in range(1,T+1):
		for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base","r",encoding="utf-8"):
			record = line.split()
			rui = float(record[2]) # 数据集评分
			Uu = mat([U[int(record[0])-1]]).reshape(1,d) # 取得这个user对应的向量 index为id-1
			Vi = mat([V[int(record[1])-1]]).reshape(1,d) # 取得这个item对应的向量 index为id-1
			ViT = mat([V[int(record[1])-1]]).reshape(d,1) # 获得这个item对应的转置向量

			gradient_Uu = -(rui-Uu*ViT)*Vi + au*Uu 
			gradient_Vi = -(rui-Uu*ViT)*Uu + av*Vi # 计算两个gradient
			U[int(record[0])-1] = Uu - r*gradient_Uu
			V[int(record[1])-1] = Vi - r*gradient_Vi # 更新当前user，item对应的向量
			
		r = r * 0.9

	# part3: calculate MAE and RMSE for different methods
	MAE = RMSE = 0.0
	for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.test","r",encoding="utf-8"):
		record = line.split()
		rui = float(mat([U[int(record[0])-1]]).reshape(1,d)*mat([V[int(record[1])-1]]).reshape(d,1))
		MAE += abs(float(record[2]) - rui)/20000
		RMSE += ((float(record[2]) - rui)**2)/20000
	RMSE = RMSE**0.5
	print("SGD_FOR_PMF：")
	print("RMSE："+str(RMSE)+"  MAE："+str(MAE))

def RSVD():
	# part1： calculate u(global average rating value), bu(bias of user), bi(bias of item)
	# u = 3.5283058961762976, bu、bi are dictionaries, key=(user_id or item_id), value = bias of (user or item)
	count = bu_count = bi_count = 1
	u = 0.0
	bu = {}
	bi = {}
	for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base","r",encoding="utf-8"):
		# line data: user id | item id | rating | timestamp
		record = line.split()
		count += 1
		u += float(record[2])
		if int(record[0]) not in bu.keys():
			bu_count = 1
			bu[int(record[0])] = [float(record[2]),bu_count]
		else:
			rate = bu[int(record[0])][0] + float(record[2])
			temp_count = bu[int(record[0])][1] + 1
			bu[int(record[0])][0] = rate
			bu[int(record[0])][1] = temp_count

		if int(record[1]) not in bi.keys():
			bi_count = 1
			bi[int(record[1])] = [float(record[2]),bi_count]
		else:
			rate = bi[int(record[1])][0] + float(record[2])
			temp_count = bi[int(record[1])][1] + 1
			bi[int(record[1])][0] = rate
			bi[int(record[1])][1] = temp_count
	u = u / count
	for key in bu.keys():
		bu[key] = (bu[key][0] - bu[key][1]*u) / bu[key][1]
	for key in bi.keys():
		bi[key] = (bi[key][0] - bi[key][1]*u) / bi[key][1]


	# part2: initialize the parameters
	au = av = Bu = Bv = 0.01 # tradeoff parameters
	r = 0.01 # learning rate
	d = 20 # number of latent dimensions
	T = 100 # iteration number
	n = 943 # user number
	m = 1682 # item number
	V = (np.random.random((m,d)) - 0.5)*0.01
	U = (np.random.random((n,d)) - 0.5)*0.01 # initialize matrixs
	for t in range(1,T+1):
		for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base","r",encoding="utf-8"):
			record = line.split()
			rui = float(record[2]) # 数据集评分
			Uu = mat([U[int(record[0])-1]]).reshape(1,d) # 取得这个user对应的向量 index为id-1
			Vi = mat([V[int(record[1])-1]]).reshape(1,d) # 取得这个item对应的向量 index为id-1
			ViT = mat([V[int(record[1])-1]]).reshape(d,1) # 获得这个item对应的转置向量

			_rui = float(u+bu[int(record[0])]+bi[int(record[1])]+mat([U[int(record[0])-1]]).reshape(1,d)*mat([V[int(record[1])-1]]).reshape(d,1))
			eui = rui - _rui
			gra_u = -eui
			gra_bu = -eui + Bu*bu[int(record[0])]
			gra_bi = -eui + Bv*bi[int(record[1])]
			gra_Uu = (-eui)*Vi + au*Uu
			gra_Vi = (-eui)*Uu + av*Vi

			u = u - r*gra_u
			bu[int(record[0])] = bu[int(record[0])] - r*gra_bu
			bi[int(record[1])] = bi[int(record[1])] - r*gra_bi
			U[int(record[0])-1] = U[int(record[0])-1] - r*gra_Uu
			V[int(record[1])-1] = V[int(record[1])-1] - r*gra_Vi
		r = r * 0.9

	# part3: calculate MAE and RMSE for different methods
	MAE = RMSE = 0.0
	for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.test","r",encoding="utf-8"):
		record = line.split()

		if int(record[0]) in bu.keys() and int(record[1]) in bi.keys():
			rui = float(u+bu[int(record[0])]+bi[int(record[1])]+mat([U[int(record[0])-1]]).reshape(1,d)*mat([V[int(record[1])-1]]).reshape(d,1))
			MAE += abs(float(record[2]) - rui)/20000
			RMSE += ((float(record[2]) - rui)**2)/20000
	RMSE = RMSE**0.5
	print("RSVD：")
	print("RMSE："+str(RMSE)+"  MAE："+str(MAE))

def PURE_SVD():
	n = 943 # user number
	m = 1682 # item number
	d = 20 # number of latent dimensions
	R1 = mat(np.zeros([n,m]),dtype=np.float32)
	for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base","r",encoding="utf-8"):
		record = line.split()
		R1[int(record[0])-1,int(record[1])-1] = float(record[2])
	R1_ZERO = np.argwhere(R1 == 0)

	sub1 = np.mean(R1,axis=1).reshape((1,n))# 计算每一行的均值
	sub2 = []
	for i in range(len(R1_ZERO)):
		B = mat(R1[R1_ZERO[i][0]])
		sub2.append(R1[R1_ZERO[i][0]].size/(R1[R1_ZERO[i][0]].size - B[B==0].size))
	for i in range(len(R1_ZERO)):
		R1[R1_ZERO[i][0],R1_ZERO[i][1]] = sub1.getA1()[R1_ZERO[i][0]]*sub2[i]# getA1()返回一个扁平（一维）的数组（ndarray）

	uag = np.mean(R1,axis=1)
	R = R1 - uag*np.ones((1,m)) 
	U,sigma,VT = la.svd(R)
	new_U = mat([U[i].getA1()[0:d] for i in range(n)])
	sigma = sorted(sigma,reverse=True)
	sigma = np.diag(sigma[0:d])
	new_VTT = mat([VT[i].getA1()[0:m] for i in range(d)])
	matrix = new_U*sigma*new_VTT

	MAE = RMSE = 0.0
	for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.test","r",encoding="utf-8"):
		record = line.split()
		rui = float(uag[int(record[0])-1]+matrix[int(record[0])-1,int(record[1])-1])
		MAE += abs(float(record[2]) - rui)/20000
		RMSE += ((float(record[2]) - rui)**2)/20000
	RMSE = RMSE**0.5
	print("PURE_SVD：")
	print("RMSE："+str(RMSE)+"  MAE："+str(MAE)) # 1.017268702219809 0.8060688157379614

if __name__ == '__main__':
	start = datetime.datetime.now()
	SGD_FOR_PMF()
	end = datetime.datetime.now()
	print("running time："+str(end-start))
	# SGD_FOR_PMF：
	# RMSE：1.8412217212710047  MAE：1.4857774990423802

	# start = datetime.datetime.now()
	# PURE_SVD()
	# end = datetime.datetime.now()
	# print("running time："+str(end-start))

	# start = datetime.datetime.now()
	# RSVD()
	# end = datetime.datetime.now()
	# print("running time："+str(end-start))