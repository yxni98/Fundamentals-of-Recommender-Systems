#!user/bin/env python3  
# -*- coding: utf-8 -*-

import numpy as np
from numpy import mat
import os,random

class SVDpp(object):
	"""docstring for SVDpp"""
	def __init__(self, base_file, test_file):
		super(SVDpp, self).__init__()
		self.base_file = base_file
		self.test_file = test_file
		self.au = self.av = self.aw = self.Bu = self.Bv = 0.01
		self.rate = 0.01
		self.d = 20
		self.T = 1
		self.u = 0.0
		self.bu = {}
		self.bi = {}
		self._II = {}
		self.n = 943
		self.m = 1682
		self.U = (np.random.random((self.n,self.d)) - 0.5)*0.01 
		self.V = (np.random.random((self.m,self.d)) - 0.5)*0.01 
		self.W = (np.random.random((self.m,self.d)) - 0.5)*0.01 # initialize matrixs
		self.explicit = self.implicit = []
		self.RMSE = self.MAE  = 0.0

	def pre_processing(self):
		data = []
		for line in open(self.base_file,"r",encoding="utf-8"):
			record = line.split()
			data.append(record)
		random.shuffle(data)
		self.explicit = data[0:int(len(data)/2)]
		self.implicit = data[int(len(data)/2):len(data)]

	def init_params(self):
		count = 0
		for record in self.explicit:
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

		for record in self.implicit:
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

	def SGD(self):
		for t in range(self.T):
			for record in self.implicit:
				if int(record[0]) in self.bu.keys() and int(record[1]) in self.bi.keys():
					Uu = mat([self.U[int(record[0])-1]]).reshape(1,self.d) # 取得这个user对应的向量 index为id-1
					Vi = mat([self.V[int(record[1])-1]]).reshape(1,self.d) # 取得这个item对应的向量 index为id-1
					ViT = mat([self.V[int(record[1])-1]]).reshape(self.d,1) # 获得这个item对应的转置向量
					_Uu = 0
					for ii in self._II[int(record[0])]:
						_Uu += mat([self.W[ii-1]]).reshape(1,self.d)
					_Uu /= (len(self._II[int(record[0])])**0.5)
					_rui = Uu*ViT + _Uu*ViT + self.bu[int(record[0])] + self.bi[int(record[1])] + self.u
					_rui = float(_rui)
					if _rui > 5:
						_rui = 5
					elif _rui < 1:
						_rui = 1
					eui  = float(record[2]) - _rui

					gra_Uu = (-eui)*Vi + self.au*Uu
					gra_Vi = (-eui)*(Uu + _Uu) + self.av*Vi
					gra_bu = -eui + self.Bu*self.bu[int(record[0])]
					gra_bi = -eui + self.Bv*self.bi[int(record[1])]
					gra_u = -eui

					self.U[int(record[0])-1] = self.U[int(record[0])-1] - self.rate*gra_Uu
					self.V[int(record[1])-1] = self.V[int(record[1])-1] - self.rate*gra_Vi
					self.bu[int(record[0])] = self.bu[int(record[0])] - self.rate*gra_bu
					self.bi[int(record[1])] = self.bi[int(record[1])] - self.rate*gra_bi
					self.u = self.u - self.rate*gra_u

					for ii in self._II[int(record[0])]:
						Wi = mat([self.W[ii-1]]).reshape(1,self.d)
						gra_Wi = ((-eui)*Vi) / (len(self._II[int(record[0])])**0.5) + self.aw*Wi
						self.W[ii-1] = self.W[ii-1] - self.rate*gra_Wi	
			self.rate = self.rate*0.9

	def RMSE_MAE(self):
		for line in open(self.test_file,"r",encoding="utf-8"):
			record = line.split()
			if int(record[0]) in self.bu.keys() and int(record[1]) in self.bi.keys() and int(record[0]) in self._II.keys():
				Uu = mat([self.U[int(record[0])-1]]).reshape(1,self.d) # 取得这个user对应的向量 index为id-1
				ViT = mat([self.V[int(record[1])-1]]).reshape(self.d,1) # 获得这个item对应的转置向量
				_Uu = 0
				for ii in self._II[int(record[0])]:
					_Uu += mat([self.W[ii-1]]).reshape(1,self.d)
				_Uu /= (len(self._II[int(record[0])])**0.5)
				_rui = Uu*ViT + _Uu*ViT + self.bu[int(record[0])] + self.bi[int(record[1])] + self.u
				_rui = float(_rui)
				if _rui > 5:
					_rui = 5
				elif _rui < 1:
					_rui = 1
				self.MAE += abs(float(record[2]) - _rui)/9430
				self.RMSE += ((float(record[2]) - _rui)**2)/9430
		self.RMSE = float(self.RMSE)
		self.MAE = float(self.MAE)
		self.RMSE = (self.RMSE)**0.5
		print("SGD for SVD++：")
		print("RMSE："+str(self.RMSE)+"  MAE："+str(self.MAE))# RMSE：0.6884531159481491  MAE：0.36846790237293425

def main():
	# load u1.base and u1.test's data into memory
	base_file = str(os.getcwd()).replace("\\","/") + "/ml-100k/ua.base"
	test_file = str(os.getcwd()).replace("\\","/") + "/ml-100k/ua.test"
	# initial the class
	svdpp = SVDpp(base_file, test_file)
	svdpp.pre_processing()
	svdpp.init_params()
	svdpp.SGD()
	svdpp.RMSE_MAE()

if __name__ == '__main__':
	main()