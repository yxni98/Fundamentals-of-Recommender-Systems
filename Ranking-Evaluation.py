#!user/bin/env python3  
# -*- coding: utf-8 -*-

import os,math

class RankingEvaluation(object):
	"""docstring for RankingEvaluation"""
	def __init__(self):
		super(RankingEvaluation, self).__init__()
		self.k = 5
		self.n = 943
		self.m = 1682
		self.bi = {}
		self.ite = {}
		self.u_i = {}
		self.u_i_k = {}
		self.i_u = {}
		self.all_items = set()
		
	def pre_processing(self):
		for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base.occf","r",encoding="utf-8"):
			record = line.split()
			self.all_items.add(int(record[1]))
			if int(record[0]) not in self.u_i.keys():
				self.u_i[int(record[0])] = [int(record[1])]
			else:
				temp_list = self.u_i[int(record[0])]
				temp_list.append(int(record[1]))
				self.u_i[int(record[0])] = temp_list # 每个用户及被打了分的物品映射 u1->[i1,i2...],u2->[i1,i2...] ...

			if int(record[1]) not in self.i_u.keys():
				self.i_u[int(record[1])] = [int(record[0])]
			else:
				temp_list = self.i_u[int(record[1])]
				temp_list.append(int(record[0]))
				self.i_u[int(record[1])] = temp_list # 每个物品及被打了分的用户映射 i1->[u1,u2...],i2->[u1,u2...] ...

	def get_train_occf(self):
		u = 44140/self.n/self.m
		for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.base.occf","r",encoding="utf-8"):
			record = line.split()
			if int(record[1]) not in self.bi.keys():
				self.bi[int(record[1])] = 1
			else:
				temp = self.bi[int(record[1])]
				temp += 1
				self.bi[int(record[1])] = temp
		for key in self.bi.keys():
			self.bi[key] = self.bi[key]/self.n - u
		dict_list = sorted(self.bi.items(),key = lambda x:x[1],reverse = True) # 预测的多项物品的ranking（降序排列）

		for line in open(str(os.getcwd()).replace("\\","/") + "/ml-100k/u1.test.occf","r",encoding="utf-8"):
			record = line.split()
			self.all_items.add(int(record[1]))
			if int(record[0]) not in self.ite.keys():
				self.ite[int(record[0])] = [int(record[1])]
			else:
				temp_list = self.ite[int(record[0])]
				temp_list.append(int(record[1]))
				self.ite[int(record[0])] = temp_list # 每个用户在test中对应的物品映射 u1->[i1,i2...],u2->[i1,i2...] ...
			
		for ui_key in self.u_i.keys():
			count = 0
			for data in dict_list:
				if data[0] not in self.u_i[ui_key]:
					if count == 0:
						self.u_i_k[ui_key] = [data[0]]
					else:
						temp_list = self.u_i_k[ui_key]
						temp_list.append(data[0])
						self.u_i_k[ui_key] = temp_list # 每个用户的前n个没推荐过的物品id（降序排列）
					count += 1

	def Pre_k(self): # 找出re前k个item中与test的交集个数，除以k，最后除以test用户数
		Pre_k = 0.0
		for key in self.ite.keys():
			Pre_k += (len(set(self.ite[key]).intersection(set(self.u_i_k[key][:self.k]))) / (self.k*len(self.ite.keys())))
		print("Pre©5: "+str(Pre_k))

	def Rec_k(self): # 找出re前k个item中与test的交集个数，除以用户对应的test中的item个数，最后除以test用户数
		Rec_k = 0.0
		for key in self.ite.keys():
			Rec_k += (len(set(self.ite[key]).intersection(set(self.u_i_k[key][:self.k]))) / (len(self.ite[key])*len(self.ite.keys())))
		print("Rec©5: "+str(Rec_k))

	def F1_k(self):
		F1_k = 0.0
		for key in self.ite.keys():
			Preuk = (len(set(self.ite[key]).intersection(set(self.u_i_k[key][:self.k]))) / (self.k))
			Recuk = (len(set(self.ite[key]).intersection(set(self.u_i_k[key][:self.k]))) / (len(self.ite[key])))
			if (Preuk + Recuk) != 0.0:
				F1_k += 2 * ((Preuk * Recuk) / (Preuk + Recuk) / len(self.ite.keys()))
		print("F1©5: "+str(F1_k))

	def NDCG_k(self):
		NDCG_k = 0.0
		for key in self.ite.keys():
			DCG_k = 0.0
			Zu = 0.0
			count = 0
			for l in range(min(self.k, len(self.ite[key]))):
				if self.u_i_k[key][l] in self.ite[key]:
					DCG_k += 1 / math.log(l+2,2)
				Zu += 1 / math.log(l+2,2)
			DCG_k /= Zu
			NDCG_k += DCG_k
		NDCG_k /= len(self.ite.keys())
		print("NDCG©5: "+str(NDCG_k))

	def one_call_k(self): # 找出test和re前k个中item交集大于等于1的用户数，最后除以test用户数
		count = 0
		for key in self.ite.keys():
			one_call_uk = 0.0
			one_call_uk += len(set(self.ite[key]).intersection(set(self.u_i_k[key][:self.k]))) 
			if one_call_uk >= 1.0:
				count += 1
		one_call_k = count/len(self.ite.keys())
		print("1-call©5: "+str(one_call_k))

	def MRR(self): # 找出第一个预测正确的item在test集合中的位置index，累加1/index，最后除以test用户数
		MRR = 0.0
		for key in self.ite.keys():
			index = 0
			for data in self.u_i_k[key]:
				index += 1
				if data in self.ite[key]:
					RRu = 1 / index
					MRR += RRu
					break
		MRR /= len(self.ite.keys())
		print("MRR: "+str(MRR))

	# 找出所以预测正确的item在test集合中的位置index，组成字典{1:[113,178...]...}
	# 对每个用户，APu等于（下标+1）/index的累加
	# 将Apu除以用户在test集合中的item个数，累加到MAP中
	def MAP(self): 
		ranking_items = {}
		for key in self.ite.keys():
			index = 0
			ranking_items[key] = []
			for data in self.u_i_k[key]:
				index += 1
				if data in self.ite[key]:
					ranking_items[key].append(index)
		MAP = 0.0
		for key in ranking_items.keys():
			APu = 0.0
			for data in ranking_items[key]:
				APu += (ranking_items[key].index(data)+1) / data
			APu /= len(self.ite[key])
			MAP += APu
		MAP /= len(self.ite.keys())
		print("MAP: "+str(MAP))
		return ranking_items
	
	# 找出所以预测正确的item在test集合中的位置index，组成字典{1:[113,178...]...}
	# 对每个用户，RPu等于 Pui / (I - Iu)的累加
	# 将Rpu除以用户在test集合中的item个数，累加到ARP中
	def ARP(self, ranking_items): 
		I = len(self.i_u.keys()) # I是base的物品集还是base.occf的物品集？（此处使用后者获得了与课件相近的结果）
		ARP = 0.0
		for key in self.ite.keys():
			Iu = len(self.u_i[key])
			RPu = 0.0
			for data in ranking_items[key]:
				RPu += data / (I - Iu)
			RPu /= len(self.ite[key])
			ARP += RPu
		ARP /= len(self.ite.keys())
		print("ARP: "+str(ARP))

	def AUC(self): # 若集合为base和test，答案为0.67，occf集合下的答案为0.8302
		R = self.u_i
		R_te = self.ite
		R_te_u_j = {}
		for key in self.ite.keys():
			if key in R.keys() and key in R_te.keys():
				R_union = set(R[key] + R_te[key])
				R_te_u_j[key] = []
				for data in self.all_items:
					if data not in R_union:
						R_te_u_j[key].append(data)
		AUC = 0.0
		for key in self.ite.keys():
			AUCu = 0
			if key in R.keys():
				for i in R[key]:
					for j in R_te_u_j[key]:
						if i in self.bi.keys() and j in self.bi.keys() and self.bi[i] > self.bi[j]:
							AUCu += 1
				AUCu /= (len(R[key]) * len(R_te_u_j[key]))
				AUC += AUCu
		AUC /= len(self.ite.keys())
		print("AUC: "+str(AUC))


def main():
	ranking = RankingEvaluation()
	ranking.pre_processing()
	ranking.get_train_occf()
	ranking.Pre_k()
	ranking.Rec_k()
	ranking.F1_k()
	ranking.NDCG_k()
	ranking.one_call_k()
	ranking.MRR()
	ranking_items = ranking.MAP()
	ranking.ARP(ranking_items)
	ranking.AUC()


if __name__ == '__main__':
	main()