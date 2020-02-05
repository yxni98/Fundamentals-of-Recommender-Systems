#!user/bin/env python3  
# -*- coding: utf-8 -*-

import numpy as np
from numpy import mat
import os,random

class MFMPC(object):
	"""docstring for MFMPC"""
	def __init__(self, base_file):
		super(MFMPC, self).__init__()
		self.base_file = base_file
		self.rate = 0.001
		self.d = 20
		self.T = 2
		self.u = 0.0
		self.bu = {}
		self.bi = {}
		self.n = 943
		self.m = 1682
		self._II = {}
		self.U = (np.random.random((self.n,self.d)) - 0.5)*0.01 
		self.V = (np.random.random((self.m,self.d)) - 0.5)*0.01 
		self.M = (np.random.random((self.m,self.d)) - 0.5)*0.01 # initialize matrixs
		self.RMSE = self.MAE  = 0.0

	def init_params(self):
		count = 0
		for record in self.base_file:
			self.u += float(record[2])
			count += 1

			if int(record[0]) not in self.bu.keys():
				self.bu[int(record[0])] = [float(record[2])]
			else:
				temp_list = self.bu[int(record[0])]
				temp_list.append(float(record[2]))
				self.bu[int(record[0])] = temp_list

			if int(record[1]) not in self.bi.keys():
				self.bi[int(record[1])] = [float(record[2])]
			else:
				temp_list = self.bi[int(record[1])]
				temp_list.append(float(record[2]))
				self.bi[int(record[1])] = temp_list

			if int(record[0]) not in self._II.keys():
				self._II[int(record[0])] = [int(record[1])]		
			else:
				temp_list = self._II[int(record[0])]
				temp_list.append(int(record[1]))
				self._II[int(record[0])] = temp_list

		if count != 0:
			self.u /= count
		for key in self.bu.keys():
			self.bu[key] = sum(self.bu[key]) / len(self.bu[key])
		for key in self.bi.keys():
			self.bi[key] = sum(self.bi[key]) / len(self.bi[key])

	def mfmpc_algorithm(self):
		for t1 in range(1):
			for record in self.base_file:
				if int(record[0]) in self.bu.keys() and int(record[1]) in self.bi.keys():
					Uu = mat([self.U[int(record[0])-1]]).reshape(1,self.d) # 取得这个user对应的向量 index为id-1
					Vi = mat([self.V[int(record[1])-1]]).reshape(1,self.d) # 取得这个item对应的向量 index为id-1
					ViT = mat([self.V[int(record[1])-1]]).reshape(self.d,1) # 获得这个item对应的转置向量
					_Uu = 0
					_m = 0.0
					for ii in self._II[int(record[0])]:
						_Uu += mat([self.M[ii-1]]).reshape(1,self.d)
						_m += 1 / (len(self._II[int(record[0])])**0.5)
					_Uu *= _m
					_rui = Uu*ViT + _Uu*ViT + self.bu[int(record[0])] + self.bi[int(record[1])] + self.u
					_rui = float(_rui)
					if _rui > 5:
						_rui = 5
					elif _rui < 1:
						_rui = 1

					eui  = float(record[2]) - _rui

					gra_Uu = (-eui)*Vi + self.rate*Uu
					gra_Vi = (-eui)*(Uu + _Uu) + self.rate*Vi
					gra_bu = -eui + self.rate*self.bu[int(record[0])]
					gra_bi = -eui + self.rate*self.bi[int(record[1])]
					gra_u = -eui

					self.U[int(record[0])-1] = self.U[int(record[0])-1] - self.rate*gra_Uu
					self.V[int(record[1])-1] = self.V[int(record[1])-1] - self.rate*gra_Vi
					self.bu[int(record[0])] = self.bu[int(record[0])] - self.rate*gra_bu
					self.bi[int(record[1])] = self.bi[int(record[1])] - self.rate*gra_bi
					self.u = self.u - self.rate*gra_u

					for ii in self._II[int(record[0])]:
						Mi = mat([self.M[ii-1]]).reshape(1,self.d)
						gra_Mi = ((-eui)*Vi) * (1 / (len(self._II[int(record[0])])**0.5)) + self.rate*Mi
						self.M[ii-1] = self.M[ii-1] - self.rate*gra_Mi
			self.rate = self.rate*0.9

	def RMSE_MAE(self, test_file):
		for record in test_file:
			if int(record[0]) in self.bu.keys() and int(record[1]) in self.bi.keys() and int(record[0]) in self._II.keys():
				Uu = mat([self.U[int(record[0])-1]]).reshape(1,self.d) # 取得这个user对应的向量 index为id-1
				Vi = mat([self.V[int(record[1])-1]]).reshape(1,self.d) # 取得这个item对应的向量 index为id-1
				ViT = mat([self.V[int(record[1])-1]]).reshape(self.d,1) # 获得这个item对应的转置向量
				_Uu = 0
				_m = 0.0
				for ii in self._II[int(record[0])]:
					_Uu += mat([self.M[ii-1]]).reshape(1,self.d)
					_m += 1 / (len(self._II[int(record[0])])**0.5)
				_Uu *= _m
				_rui = Uu*ViT + _Uu*ViT + self.bu[int(record[0])] + self.bi[int(record[1])] + self.u
				_rui = float(_rui)
				if _rui > 5:
					_rui = 5
				elif _rui < 1:
					_rui = 1
				self.MAE += abs(float(record[2]) - _rui)/20000
				self.RMSE += ((float(record[2]) - _rui)**2)/20000
		self.RMSE = float(self.RMSE)
		self.MAE = float(self.MAE)
		self.RMSE = (self.RMSE)**0.5
		print("SGD for MFMPC++：")
		print("RMSE："+str(self.RMSE)+"  MAE："+str(self.MAE))
		self.RMSE = self.MAE  = 0.0

def main():
	chosen_file = "C:/Users/lenovo/Desktop/RS/ml-100k/u.data"
	count = 0
	file = []
	inner_file = []
	for line in open(chosen_file, encoding="utf-8"):
		record = line.split()
		inner_file.append(record)
		count += 1
		if count%20000 == 0:
			file.append(inner_file)
			inner_file = []
	base_file = file[0] + file[1] + file[2] + file[3]
	test_file = file[4]
	print("test time 1") # RMSE：0.9577858189370168  MAE：0.7562895836628751 # RMSE：0.949201514496287  MAE：0.7497721288353782
	mfmpc = MFMPC(base_file) # RMSE：0.9453801920232396  MAE：0.7469493187910594
	mfmpc.init_params()
	# mfmpc.mfmpc_algorithm()
	# mfmpc.RMSE_MAE(test_file)

	# base_file = file[0] + file[1] + file[2] + file[4]
	# test_file = file[3]
	# print("test time 2") # RMSE：0.9583875408090251  MAE：0.7531570481622692
	# mfmpc = MFMPC(base_file)
	# mfmpc.init_params()
	# mfmpc.mfmpc_algorithm()
	# mfmpc.RMSE_MAE(test_file)

	# base_file = file[0] + file[1] + file[4] + file[3]
	# test_file = file[2]
	# print("test time 3")
	# mfmpc = MFMPC(base_file)
	# mfmpc.init_params()
	# mfmpc.mfmpc_algorithm()
	# mfmpc.RMSE_MAE(test_file)

	# base_file = file[0] + file[4] + file[2] + file[3]
	# test_file = file[1]
	# print("test time 4")
	# mfmpc = MFMPC(base_file)
	# mfmpc.init_params()
	# mfmpc.mfmpc_algorithm()
	# mfmpc.RMSE_MAE(test_file)

	# base_file = file[4] + file[1] + file[2] + file[3]
	# test_file = file[0]
	# print("test time 5")
	# mfmpc = MFMPC(base_file)
	# mfmpc.init_params()
	# mfmpc.mfmpc_algorithm()
	# mfmpc.RMSE_MAE(test_file)

if __name__ == '__main__':
	main()
	