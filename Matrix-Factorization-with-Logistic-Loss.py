#!user/bin/env python3  
# -*- coding: utf-8 -*-

import time,os,random,math

class MFLogLoss(object):
	"""docstring for MFLogLoss"""
	def __init__(self):
		super(MFLogLoss, self).__init__()
		self.n = 943
		self.m = 1682
		self.p = 3
		self.d = 20
		self.y = 0.01
		self.au = 0.001
		self.av = 0.001
		self.Bu = 0.001
		self.Bv = 0.001
		self.T = 1
		self.bu = {}
		self.bi = {}
		self.observed_item = []
		self.P = []
		self.whole_A = []
		self.A = []
		self.P_size = 0
		self.Vi = {}
		self.Uu = {}
		self.Iu = {}
		self.Ui = {}
		self.k = 5
		self.re = {}
		self.test_Iu = {}
		for line in open("ml-100k/u1.base.occf","r",encoding="utf-8"):
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
			Uuk = []
			for x in range(self.d):
				Vik.append((random.random()-0.5)*0.01)
				Uuk.append((random.random()-0.5)*0.01)
			self.Vi[i] = Vik
			self.Uu[i] = Uuk

		self.observed_item = set(self.observed_item)

	def training(self):
		for t in range(self.T): 
			self.A = random.sample(self.whole_A, self.p*self.P_size)
			union_pairs = self.A + random.sample(self.P, self.P_size)
			union_pairs = random.sample(union_pairs, len(union_pairs))
			for user,item in union_pairs:
				_rui = self.bu[user] + self.bi[item]
				for i in range(self.d):
					_rui += (self.Uu[user][i] * self.Vi[item][i])
				if (user,item) in self.P:
					eui = (1 / (1 + math.exp(_rui)))
				else: 
					eui = (-1 / (1 + math.exp(-_rui)))
				gra_bu = -eui + self.Bu*self.bu[user]
				gra_bi = -eui + self.Bv*self.bi[item]
				gra_Vi = [-eui*self.Uu[user][x] + self.av*self.Vi[item][x] for x in range(self.d)]
				gra_Uu = [-eui*self.Vi[item][x] + self.au*self.Uu[user][x] for x in range(self.d)]
				
				self.bu[user] -= self.y*gra_bu
				self.bi[item] -= self.y*gra_bi
				for x in range(self.d):
					self.Vi[item][x] -= self.y*gra_Vi[x]
					self.Uu[user][x] -= self.y*gra_Uu[x]
			print(t+1)

	def top_k(self):
		for line in open("ml-100k/u1.test.occf","r",encoding="utf-8"):
			record = line.split()
			if(int(record[0])) not in self.test_Iu.keys():
				self.test_Iu[int(record[0])] = (int(record[1]),)
			else:
				self.test_Iu[int(record[0])] += (int(record[1]),)

		for user in self.test_Iu.keys():
			ranking_list = {}
			for item in (self.observed_item - set(self.Iu[user])):
				if user in self.bu.keys() and item in self.bi.keys():		
					_rui = self.bu[user] + self.bi[item]					
					for i in range(self.d):
						_rui += (self.Uu[user][i] * self.Vi[item][i])
					ranking_list[item] = _rui
			top_k_pairs = sorted(ranking_list.items(),key = lambda x:x[1],reverse = True)[:self.k]
			self.re[user] = [pair[0] for pair in top_k_pairs]

	def Pre_k(self): # 找出re前k个item中与test的交集个数，除以k，最后除以test用户数
		Pre_k = 0.0 
		for key in self.test_Iu.keys():
			Pre_k += (len(set(self.test_Iu[key]).intersection(set(self.re[key]))) / (self.k*len(self.test_Iu.keys())))
		print("Pre_5: "+str(Pre_k))

	def Rec_k(self): # 找出re前k个item中与test的交集个数，除以用户对应的test中的item个数，最后除以test用户数
		Rec_k = 0.0
		for key in self.test_Iu.keys():
			Rec_k += (len(set(self.test_Iu[key]).intersection(set(self.re[key]))) / (len(self.test_Iu[key])*len(self.test_Iu.keys())))
		print("Rec_5: "+str(Rec_k))

def main():
	start = time.time()
	mflogloss = MFLogLoss()
	mflogloss.training()
	mflogloss.top_k()
	mflogloss.Pre_k()
	mflogloss.Rec_k()
	end = time.time()
	print("finished in "+str(end-start)+" s")

if __name__ == '__main__':
	main()