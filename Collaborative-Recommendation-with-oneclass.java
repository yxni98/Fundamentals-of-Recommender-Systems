package com.rs;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;

class OPC {
	double γ = 0.01;
	double λ = 0.001; //0.001, 0.01, 0.1
	int d = 20;
	int T = 50;
	int n = 943;
	int m = 1682;	
	double[][] Uu = new double[n+1][d];
	double[][] Vi = new double[m+1][d];
	double[][] Mi = new double[m+1][d];
	double[][] TrainingData = new double[80001][3];
	double[][] TestData= new double[20001][3];
	int R = 80000;
	int Rte = 20000;
	double u = 0.0;
	double[] bu = new double[n+1];
	double[] bi = new double[m+1];
	double[][][] AllData = new double[5][20000][3];
	HashMap<Integer,HashSet<Integer>> Ui = new HashMap<Integer,HashSet<Integer>>();
	HashMap<Integer,HashSet<Integer>> Iu = new HashMap<Integer,HashSet<Integer>>();
	public OPC(int[] PartIndex) throws NumberFormatException, IOException {
		// TODO Auto-generated constructor stub
		for (int i = 1; i <= n; i++) {
			for (int k = 0; k < d; k++) {
				Uu[i][k] = (Math.random() - 0.5) * 0.01;
			}
		}	
		for (int i = 1; i <= m; i++) {
			for (int k = 0; k < d; k++) {
				Vi[i][k] = (Math.random() - 0.5) * 0.01;
				Mi[i][k] = (Math.random() - 0.5) * 0.01;
			}
		}
		String ReadContent = null;
		BufferedReader Reader = new BufferedReader(new FileReader(new File("C:/Users/lenovo/Desktop/RS/ml-100k/u.data")));
		HashSet<Integer> set = new HashSet<>();
		int count = 0;
		int PartCount = 0;
		while((ReadContent=Reader.readLine())!=null){
			String[] row = ReadContent.split("\\s+");
			if((count)%20000==0&&count!=0){
				PartCount++;
				count = 0;
			}
			for(int i = 0; i < 3; i++){
				AllData[PartCount][count][i] = Double.valueOf(row[i]);
			}
			count++;		
		}
		Reader.close();
		for(int c = 0; c < 4; c++){
			int ChosenPart = PartIndex[c];
			for(int i = 0; i < 20000; i++){
				if(!Iu.containsKey((int)AllData[ChosenPart][i][0])){
					HashSet<Integer> NewSet = new HashSet<>();
					NewSet.add((int)AllData[ChosenPart][i][1]);
					Iu.put((int)AllData[ChosenPart][i][0], NewSet);
				}
				else{
					set = Iu.get((int)AllData[ChosenPart][i][0]);
					set.add((int)AllData[ChosenPart][i][1]);
					Iu.put((int)AllData[ChosenPart][i][0], set);
				}
				
				if(!Ui.containsKey((int)AllData[ChosenPart][i][1])){
					HashSet<Integer> NewSet = new HashSet<>();
					NewSet.add((int)AllData[ChosenPart][i][0]);
					Ui.put((int)AllData[ChosenPart][i][1], NewSet);
				}
				else{
					set = Ui.get((int)AllData[ChosenPart][i][1]);
					set.add((int)AllData[ChosenPart][i][0]);
					Ui.put((int)AllData[ChosenPart][i][1], set);
				}
				u += (double)AllData[ChosenPart][i][2];
				bu[(int)AllData[ChosenPart][i][0]] += (double)AllData[ChosenPart][i][2];
				bi[(int)AllData[ChosenPart][i][1]] += (double)AllData[ChosenPart][i][2];
			}
		}		
		u /= 80000;
		for (int i = 1; i <= n; i++) {
			if(Iu.containsKey(i))
			bu[i] = (bu[i] - Iu.get(i).size()*u)/(double)Iu.get(i).size();
		}
		for (int i = 1; i <= m; i++) {
			if(Ui.containsKey(i))
				bi[i] = (bi[i] - Ui.get(i).size()*u)/(double)Ui.get(i).size();
		}
	}	
	
	void Training() {
		int ChosenIndex;
		double _rui;
		int user;
		int item;
		double rui;
		double eui;
		double Grabu;
		double Grabi;
		double Grau;
		for(int t = 0; t < T; t++){
			for(int t2 = 0; t2 < 80000; t2++){
				ChosenIndex = 1 + (int)(Math.random() * (80000));
				double[] ChosenRating = AllData[ChosenIndex/20000][ChosenIndex%20000];
				user = (int)ChosenRating[0];
				item = (int)ChosenRating[1];
				rui = ChosenRating[2];
				_rui = bu[user] + bi[item] + u;
				double LengthOfIu_i = Iu.get(user).size();
				double[] UuMPC = new double[d];
				for (Integer i : Iu.get(user)) {
					if(i==item){
						LengthOfIu_i--;
					}
					else{
						for(int k = 0; k < d; k++){
							UuMPC[k] += Mi[i][k];
						}
					}
				}
				LengthOfIu_i = Math.sqrt(LengthOfIu_i);
				for(int k = 0; k < d; k++){
					UuMPC[k] /= LengthOfIu_i;
				}
				for(int k = 0; k < d; k++){
					_rui += (Uu[user][k]*Vi[item][k] + UuMPC[k]*Vi[item][k]);
				}
				eui = rui - _rui;
				
				double[] GraUu = new double[d];
				double[] GraVi = new double[d];			
				for(int k = 0; k < d; k++){
					GraUu[k] = -eui*Vi[item][k] + λ*Uu[user][k];
					GraVi[k] = -eui*(Uu[user][k] + UuMPC[k]) + λ*Vi[item][k];
				}
				
				Grabu = -eui + λ*bu[user];
				Grabi = -eui + λ*bi[item];
				Grau = -eui;
				
				for (Integer i : Iu.get(user)) {
					if(i!=item){
						for(int k = 0; k < d; k++){
							Mi[i][k] -= γ*(-eui*Vi[item][k]/LengthOfIu_i + λ*Mi[i][k]);
						}
					}
				}
				
				for(int k = 0; k < d; k++){
					Uu[user][k] -= γ*GraUu[k];
					Vi[item][k] -= γ*GraVi[k];
				}		
				
				bu[user] -= γ*Grabu;
				bi[item] -= γ*Grabi;
				u -= γ*Grau;
			}
			γ *= 0.9;
		}
	}
	
	double MAE(int[] PartIndex) {
		double MAE = 0.0;
		int ChosenPart = PartIndex[4];
		int user;
		int item;
		double rui = 0.0;
		double _rui = 0.0;
		for(int r = 0; r < 20000; r++){
			double[] ChosenRating = AllData[ChosenPart][r];
			user = (int)ChosenRating[0];
			item = (int)ChosenRating[1];
			rui = ChosenRating[2];
			_rui = bu[user] + bi[item] + u;
			double LengthOfIu_i = Iu.get(user).size();
			double[] UuMPC = new double[d];
			for (Integer i : Iu.get(user)) {
				if(i==item){
					LengthOfIu_i--;
				}
				else{
					for(int k = 0; k < d; k++){
						UuMPC[k] += Mi[i][k];
					}
				}
			}
			LengthOfIu_i = Math.sqrt(LengthOfIu_i);
			for(int k = 0; k < d; k++){
				UuMPC[k] /= LengthOfIu_i;
			}
			for(int k = 0; k < d; k++){
				_rui += (Uu[user][k]*Vi[item][k] + UuMPC[k]*Vi[item][k]);
			}
			if(_rui>5){
				_rui = 5;
			}
			else if(_rui<1){
				_rui = 1;
			}
			MAE += Math.abs(rui - _rui) / 20000;
		}	
		return MAE;
	}
	
	double RMSE(int[] PartIndex) {
		double RMSE = 0.0;
		int ChosenPart = PartIndex[4];
		int user;
		int item;
		double rui = 0.0;
		double _rui = 0.0;
		for(int r = 0; r < 20000; r++){
			double[] ChosenRating = AllData[ChosenPart][r];
			user = (int)ChosenRating[0];
			item = (int)ChosenRating[1];
			rui = ChosenRating[2];
			_rui = bu[user] + bi[item] + u;
			double LengthOfIu_i = Iu.get(user).size();
			double[] UuMPC = new double[d];
			for (Integer i : Iu.get(user)) {
				if(i==item){
					LengthOfIu_i--;
				}
				else{
					for(int k = 0; k < d; k++){
						UuMPC[k] += Mi[i][k];
					}
				}
			}
			LengthOfIu_i = Math.sqrt(LengthOfIu_i);
			for(int k = 0; k < d; k++){
				UuMPC[k] /= LengthOfIu_i;
			}
			for(int k = 0; k < d; k++){
				_rui += (Uu[user][k]*Vi[item][k] + UuMPC[k]*Vi[item][k]);
			}
			if(_rui>5){
				_rui = 5;
			}
			else if(_rui<1){
				_rui = 1;
			}
			RMSE += (rui - _rui)*(rui - _rui) / 20000;
		}	
		RMSE = Math.sqrt(RMSE);
		return RMSE;
	}
}

public class MFOPC {
	public static void main(String[] args) throws IOException {
		double MAE = 0.0;
		double RMSE = 0.0;
		long startTime = System.currentTimeMillis();
		for(int i = 0; i < 5; i++){
			int[] PartIndex = new int[5];
			for(int j = 0; j < 5; j++){
				PartIndex[j] = (i+j)%5;
			}
			OPC MFOPC = new OPC(PartIndex);
			MFOPC.Training();
			MAE += MFOPC.MAE(PartIndex);
			RMSE += MFOPC.RMSE(PartIndex);
		}
		MAE /= 5;
		RMSE /= 5;
		System.out.println("MAE: "+MAE);
		System.out.println("RMSE: "+RMSE);
		long endTime = System.currentTimeMillis();
		System.out.println("程序运行时间： "+(endTime-startTime)+"ms");
	}
}