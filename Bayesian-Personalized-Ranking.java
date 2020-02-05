package com.rs;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Map;
import java.util.Map.Entry;
import java.util.TreeMap;

import com.sun.org.apache.xpath.internal.axes.OneStepIterator;

class Algorithm{
	public Algorithm() throws IOException {
		// initialize part
		File TrainingData = new File("C:/Users/lenovo/Desktop/RS/ml-100k/u1.base.occf");
		File TestData = new File("C:/Users/lenovo/Desktop/RS/ml-100k/u1.test.occf");
		BufferedReader reader = new BufferedReader(new FileReader(TrainingData));
		String ReadContent = null;
		HashMap<Integer,HashSet<Integer>> Ui = new HashMap<Integer,HashSet<Integer>>();
		HashMap<Integer,HashSet<Integer>> Iu = new HashMap<Integer,HashSet<Integer>>();
		HashSet<Integer> set = new HashSet<>();
		int R = 44140;
		int d = 20;
		int T = 500;
		int K = 5;
		double n = 943;
		double m = 1682;
		double y = 0.01;
		double au = 0.01;
		double av = 0.01;
		double Bv = 0.01;
		int[][] pairs = new int[R][2];
		double[][] Vi = new double[(int)m][d];
		for (int i = 0; i < m; i++) {
			for (int k = 0; k < d; k++) {
				Vi[i][k] = (Math.random() - 0.5) * 0.01;
			}
		}
		double[][] Uu = new double[(int)n][d];
		for (int u = 0; u < n; u++) {
			for (int k = 0; k < d; k++) {
				Uu[u][k] = (Math.random() - 0.5) * 0.01;
			}
		}
		int count = 0;
		while((ReadContent=reader.readLine())!=null){
			String[] row = ReadContent.split("\\s+");
			if(!Iu.containsKey(Integer.valueOf(row[0]))){
				HashSet<Integer> NewSet = new HashSet<>();
				NewSet.add(Integer.valueOf(row[1]));
				Iu.put(Integer.valueOf(row[0]), NewSet);
			}
			else{
				set = Iu.get(Integer.valueOf(row[0]));
				set.add(Integer.valueOf(row[1]));
				Iu.put(Integer.valueOf(row[0]), set);
			}
			
			if(!Ui.containsKey(Integer.valueOf(row[1]))){
				HashSet<Integer> NewSet = new HashSet<>();
				NewSet.add(Integer.valueOf(row[0]));
				Ui.put(Integer.valueOf(row[1]), NewSet);
			}
			else{
				set = Ui.get(Integer.valueOf(row[1]));
				set.add(Integer.valueOf(row[0]));
				Ui.put(Integer.valueOf(row[1]), set);
			}
			pairs[count][0] = Integer.valueOf(row[0]);
			pairs[count][1] = Integer.valueOf(row[1]);
			count++;
		}
		reader.close();
		double _u = (double)R / (double)n / (double)m;
		double[] bi = new double[(int)m+1];
		Iterator iter = Ui.entrySet().iterator();	
			while (iter.hasNext()) {
			Map.Entry entry = (Map.Entry) iter.next();	
			Object item = entry.getKey();		
			Object users = entry.getValue();
			bi[(int)item] = ((HashSet)users).size()/n - _u;
		}
			
			
		// training part
		int RandomU = 0;
		int RandomI = 0;
		int RandomJ = 0;
		double rui = 0.0;
		double ruj = 0.0;
		double sigma = 0.0;
		HashSet<Integer> ItemSet = null;
		for (int i = 0; i < T; i++) {
			for (int j = 0; j < R; j++) {
				int[] RandomPair = pairs[(int)(Math.random() * (R))];
				RandomU = RandomPair[0];
				RandomI = RandomPair[1];
				ItemSet = Iu.get(RandomU);	
				RandomJ = 0;
				
				while(ItemSet.contains(RandomJ)||RandomJ==0||bi[RandomJ]==0.0){
					RandomJ = 1 + (int)(Math.random() * (m)); //获取随机物品j不属于Iu且保证它在item总集合中
				}
				
				
				rui = bi[RandomI];
				ruj = bi[RandomJ];
				for (int inner_i = 0; inner_i < d; inner_i++) {
					rui += Uu[RandomU-1][inner_i] * Vi[RandomI-1][inner_i];
					ruj += Uu[RandomU-1][inner_i] * Vi[RandomJ-1][inner_i];
				}
							
				sigma = 1.0 / (1 + Math.pow(Math.E, rui-ruj));
							
				for (int inner_i = 0; inner_i < d; inner_i++) {
					Uu[RandomU-1][inner_i] -= y * ((Vi[RandomI-1][inner_i] - Vi[RandomJ-1][inner_i])*(-sigma) + au*Uu[RandomU-1][inner_i]);
					Vi[RandomI-1][inner_i] -= y * (Uu[RandomU-1][inner_i]*(-sigma) + av*Vi[RandomI-1][inner_i]);
					Vi[RandomJ-1][inner_i] -= y * (Uu[RandomU-1][inner_i]*sigma + av*Vi[RandomJ-1][inner_i]);
				}
			
				bi[RandomI] -= y*(-sigma + Bv*bi[RandomI]);				
				bi[RandomJ] -= y*(sigma + Bv*bi[RandomJ]);					
				
			}	
		}
		
		// get top-k part
		int user = 0;
		HashSet<Integer> items = null;
		double single_rating = 0.0;	
		TreeMap<Integer,Double> predicted_rating = new TreeMap<Integer,Double>();
		HashMap<Integer,HashSet<Integer>> top_k = new HashMap<Integer,HashSet<Integer>>();
		iter = Iu.entrySet().iterator();	
		while (iter.hasNext()) {
			Map.Entry entry = (Map.Entry) iter.next();	
			user = (int)entry.getKey();
			items = (HashSet)entry.getValue();
			predicted_rating.clear();
			for (int item = 1; item <= m; item++) {
				if(!items.contains(item)){
					single_rating = bi[item];
					for (int inner_i = 0; inner_i < d; inner_i++) {
						single_rating += Uu[user-1][inner_i] * Vi[item-1][inner_i];
					}
					predicted_rating.put(item, single_rating);
				}
			}
			items.clear(); 
			ValueComparator bvc =  new ValueComparator(predicted_rating); 
			TreeMap<Integer,Double> sorted_map = new TreeMap<Integer,Double>(bvc);
			sorted_map.putAll(predicted_rating); 
			Iterator ii = sorted_map.entrySet().iterator();	
			count = 0;
			while(ii.hasNext()){
				Map.Entry item_entry = (Map.Entry) ii.next();	
				items.add((int)item_entry.getKey());
				count++;
				if(count==K)
					break;
			}
			top_k.put(user, items);
		}
		
		
		// read data from test-data
		HashMap<Integer,HashSet<Integer>> TestIu = new HashMap<Integer,HashSet<Integer>>();
		reader = new BufferedReader(new FileReader(TestData));
		while((ReadContent=reader.readLine())!=null){
			String[] row = ReadContent.split("\\s+");
			if(!TestIu.containsKey(Integer.valueOf(row[0]))){
				HashSet<Integer> NewSet = new HashSet<>();
				NewSet.add(Integer.valueOf(row[1]));
				TestIu.put(Integer.valueOf(row[0]), NewSet);
			}
			else{
				set = TestIu.get(Integer.valueOf(row[0]));
				set.add(Integer.valueOf(row[1]));
				TestIu.put(Integer.valueOf(row[0]), set);
			}
		}
		reader.close();
		
		
		// get Pre@5 and Rec@5
		double Pre_k = 0.0;
		double Rec_k = 0.0;
		iter = TestIu.entrySet().iterator();
		HashSet RetainSet = null;
		double RetainSetNumber = 0;
		double AllUserNo = TestIu.size();
		double TestItemNo = 0;
		while(iter.hasNext()){
			Map.Entry entry = (Map.Entry) iter.next();
			user = (int)entry.getKey();
			items = (HashSet)entry.getValue();
			TestItemNo = items.size();
			items.retainAll(top_k.get(user));
			RetainSetNumber = items.size();
			Pre_k += RetainSetNumber / (double)K / AllUserNo;
			Rec_k += RetainSetNumber / TestItemNo / AllUserNo;
		}
		System.out.println(Pre_k);
		System.out.println(Rec_k);
	}
	
	// Comparator: used for descending order sort
	class ValueComparator implements Comparator<Integer> {  
		  
	    Map<Integer,Double> base;  
	    public ValueComparator(Map<Integer,Double> base) {  
	        this.base = base;  
	    }  
	  
	    public int compare(Integer a, Integer b) {  
	        if (base.get(a) >= base.get(b)) {  
	            return -1;  
	        } else {  
	            return 1;  
	        } 
	    }  
	}  
}



public class BPR {
	public static void main(String[] args) throws IOException {
		long startTime=System.currentTimeMillis();
		new Algorithm();
		long endTime=System.currentTimeMillis();
		System.out.println("程序运行时间： "+(endTime-startTime)+"ms");
	}
}