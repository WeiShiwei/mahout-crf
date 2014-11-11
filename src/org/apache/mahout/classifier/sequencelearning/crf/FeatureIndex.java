package org.apache.mahout.classifier.sequencelearning.crf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

public class FeatureIndex {

	class Pair{
		int ID=0;
		int Freq=0;
		public Pair(int id,int freq){
			ID=id;
			Freq=freq;
		}
	}
	private int ysize=0;//隐藏状态集合的大小，在IndexingHiddenState函数中更新设置
	private int maxid=0;//特征函数的个数，在IndexingFeatureIndex函数中更新设置
	//注意：如果Writable类型作为Map容器的类型会出错的，遍历的时候全部是最后一个元素，但我不知道为什么
	private Map<String, Pair> FeatureIndexMap = new HashMap<String, Pair>();
	private Map<String, Integer> HStateIndexMap = new HashMap<String, Integer>();
	
	public void IndexingHStateIndex(Set<String> HiddenStateSet){
		System.out.print("FeatureIndex::IndexingHStateIndex()=");
		for(String hiddenState : HiddenStateSet){
			System.out.print(hiddenState+" ");
		}
		System.out.println();
		
		//HidStateIndexMap和ysize
		ysize=HiddenStateSet.size();
		int id=0;
		for(String hiddenState : HiddenStateSet){
			if(hiddenState!=""){
				HStateIndexMap.put(hiddenState, id);
				id++;
			}
		}
	}
	public void IndexingFeatureIndex(TaggerImpl tagger){
		for(ArrayList<String> featureList : tagger.xStr){
			for(String feature : featureList){
				if(!FeatureIndexMap.containsKey(feature)){
					Pair idFreq=new Pair(maxid,1);
					FeatureIndexMap.put(feature,idFreq);
					if(feature.startsWith("U")){
						maxid+=ysize;
					}else{
						maxid+=ysize*ysize;
					}
				}else{
					FeatureIndexMap.get(feature).Freq++;
				}
			}
		}
		
	}
	
	public void Register(TaggerImpl tagger){
		for(ArrayList<String> featurelist : tagger.xStr){
			ArrayList<Integer> fvector=new ArrayList<Integer>();
			for(String feature:featurelist){
				fvector.add(FeatureIndexMap.get(feature).ID);
			}
			tagger.x.add(fvector);///
		}
		
		for(String hiddenstate : tagger.answerStr){
			tagger.answer.add(HStateIndexMap.get(hiddenstate));///
		}
		tagger.xsize=tagger.answerStr.size();///
		tagger.ysize=ysize;///
	}
	
	public int getMaxID(){
		return maxid;
	}
	
}
