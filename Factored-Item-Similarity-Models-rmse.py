#!user/bin/env python3  
# -*- coding: utf-8 -*-

import time,os,random

class FISM(object):
	"""docstring for FISM"""
	def __init__(self, T):
		super(FISM, self).__init__()
		self.a = 0.5
		self.r = 0.01
		self.p = 3
		self.d = 20
		self.av = 0.001
		self.aw = 0.001
		self.Bu = 0.001
		self.Bv = 0.001
		self.T = T
		self.bu = {}
		self.bi = {}
		self.u = 0
		self.n = 943
		self.m = 1682
		self.Vi = {}
		self.Wi = {}
		self.P = []
		self.whole_A = []
		self.A = []
		self.P_size = 0
		self.test_Iu = {}
		self.Iu = {}
		self.Ui = {}
		self.k = 5
		self.re = {}
		self.observed_item = []
		for line in open("G:/RS/ml-100k/u1.base.occf","r",encoding="utf-8"):
			record = line.split()
			if(int(record[0])) not in self.bu.keys():
				self.bu[int(record[0])] = (int(record[1]),)
				self.Iu[int(record[0])] = (int(record[1]),)
			else:
				self.bu[int(record[0])] += (int(record[1]),)
				self.Iu[int(record[0])] += (int(record[1]),)
			if(int(record[1])) not in self.bi.keys():
				self.bi[int(record[1])] = (int(record[0]),)
				self.Ui[int(record[1])] = (int(record[0]),)
			else:
				self.bi[int(record[1])] += (int(record[0]),)
				self.Ui[int(record[1])] += (int(record[0]),)
			if int(record[1]) not in self.observed_item:
				self.observed_item.append(int(record[1]))
			self.P.append((int(record[0]),int(record[1])))
		self.P_size = len(self.P)
		for user in self.Iu.keys():
			for item in self.Ui.keys():
				self.whole_A.append((user,item))
		self.whole_A = list(set(self.whole_A) - set(self.P))
		self.u = len(self.P) / self.n / self.m
		for user in self.bu.keys():
			self.bu[user] = len(self.bu[user]) / self.m - self.u
		for item in self.bi.keys():
			self.bi[item] = len(self.bi[item]) / self.n - self.u
		for i in range(1,self.m+1):
			Vik = []
			Wik = []
			for x in range(self.d):
				Vik.append((random.random()-0.5)*0.01)
				Wik.append((random.random()-0.5)*0.01)
			self.Vi[i] = Vik
			self.Wi[i] = Wik

		self.observed_item = set(self.observed_item)

	def training(self):
		print("training: T = "+str(self.T))
		for t in range(self.T): 
			self.A = random.sample(self.whole_A, self.p*self.P_size)
			union_pairs = self.A + random.sample(self.P, self.P_size)
			union_pairs = random.sample(union_pairs, len(union_pairs))
			for user,item in union_pairs:
				item_list = list(self.Iu[user])
				if item in item_list:
					item_list.remove(item)		
				if len(item_list)==0:
					continue						
				Uu = [0] * self.d
				for i in item_list:
					for k in range(self.d):
						Uu[k] += self.Wi[i][k]
				_rui = self.bu[user] + self.bi[item]
				Iu_i_a = len(item_list)**self.a
				for i in range(self.d):
					Uu[i] /= Iu_i_a
					_rui += (Uu[i] * self.Vi[item][i])
				if (user,item) in self.P:
					rui = 1
				else:
					rui = 0
				eui = rui - _rui
				self.bu[user] = self.bu[user] - self.r * (-eui + self.Bu*self.bu[user])
				self.bi[item] = self.bi[item] - self.r * (-eui + self.Bv*self.bi[item])
				self.Vi[item] = [self.Vi[item][x] - self.r*((-eui)*Uu[x]+self.av*self.Vi[item][x]) for x in range(self.d)]
				for i in item_list:
					self.Wi[i] = [self.Wi[i][x] - self.r*((-eui)*self.Vi[item][x]/Iu_i_a+self.aw*self.Wi[i][x]) for x in range(self.d)]
	
	def top_k(self):
		for line in open("G:/RS/ml-100k/u1.test.occf","r",encoding="utf-8"):
			record = line.split()
			if(int(record[0])) not in self.test_Iu.keys():
				self.test_Iu[int(record[0])] = (int(record[1]),)
			else:
				self.test_Iu[int(record[0])] += (int(record[1]),)

		for user in self.test_Iu.keys():
			ranking_list = {}
			item_list = list(self.Iu[user])
			if len(item_list)==0:
				continue
			else:
				Iu_i_a = len(item_list)**self.a
			for item in (self.observed_item - set(self.Iu[user])):
				if user in self.bu.keys() and item in self.bi.keys():		
					_rui = self.bu[user] + self.bi[item]					
					Uu = [0] * self.d
					for i in item_list:
						for k in range(self.d):
							Uu[k] += self.Wi[i][k]
					for i in range(self.d):
						Uu[i] /= Iu_i_a
						_rui += (Uu[i] * self.Vi[item][i])
					ranking_list[item] = _rui
			top_k_pairs = sorted(ranking_list.items(),key = lambda x:x[1],reverse = True)[:self.k]
			self.re[user] = [pair[0] for pair in top_k_pairs]

	def Pre_k(self): # 找出re前k个item中与test的交集个数，除以k，最后除以test用户数
		Pre_k = 0.0
		for key in self.test_Iu.keys():
			Pre_k += (len(set(self.test_Iu[key]).intersection(set(self.re[key]))) / (self.k*len(self.test_Iu.keys())))
		return Pre_k
		

	def Rec_k(self): # 找出re前k个item中与test的交集个数，除以用户对应的test中的item个数，最后除以test用户数
		Rec_k = 0.0
		for key in self.test_Iu.keys():
			Rec_k += (len(set(self.test_Iu[key]).intersection(set(self.re[key]))) / (len(self.test_Iu[key])*len(self.test_Iu.keys())))
		return Rec_k
		print("Rec_5: "+str(Rec_k))

def main():
	for x in range(1,10):
		fism = FISM(10*x)
		fism.training()
		fism.top_k()
		Pre_k = fism.Pre_k()
		Rec_k = fism.Rec_k()
		print("Pre_5: "+str(Pre_k)+"  Rec_5: "+str(Rec_k))

if __name__ == '__main__':
	main()