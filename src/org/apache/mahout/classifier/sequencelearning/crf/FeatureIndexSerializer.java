package org.apache.mahout.classifier.sequencelearning.crf;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

/**
* Utils for serializing Writable parts of HmmModel (that means without hidden state names and so on)
*/
final class FeatureIndexSerializer {

	private FeatureIndexSerializer() {
	}

	static void serialize(FeatureIndexer featureIndexer, DataOutput output) throws IOException {
		Map<String, Integer> featureIndexMap=featureIndexer.getFeatureIndexMap();
		Map<Integer, Integer> indexFreqMap=featureIndexer.getIndexFreqMap();
		Map<String, Integer> hiddenStateIndexMap=featureIndexer.getHiddenStateIndexMap();
		
		MapWritable featureIndexMapWritable=new MapWritable();
		Set<String> keys=featureIndexMap.keySet();//
		Iterator<String> it=keys.iterator();
		while(it.hasNext()){
			String key=it.next();
			Integer value=featureIndexMap.get(key);
			//System.out.println("key="+key+"	value="+value);
			featureIndexMapWritable.put(new Text(key), new IntWritable(value));
		}
		featureIndexMapWritable.write(output);
		
		MapWritable indexFreqMapWritable=new MapWritable();//
		Set<Integer> keys_1=indexFreqMap.keySet();
		Iterator<Integer> it_1=keys_1.iterator();
		while(it_1.hasNext()){
			Integer key=it_1.next();
			Integer value=indexFreqMap.get(key);
			//System.out.println("key="+key+"	value="+value);
			indexFreqMapWritable.put(new IntWritable(key), new IntWritable(value));
		}
		indexFreqMapWritable.write(output);
		
		MapWritable hiddenStateIndexMapWritable=new MapWritable();//
		Set<String> keys_2=hiddenStateIndexMap.keySet();
		Iterator<String> it_2=keys_2.iterator();
		while(it_2.hasNext()){
			String key=it_2.next();
			Integer value=hiddenStateIndexMap.get(key);
			//System.out.println("key="+key+"	value="+value);
			hiddenStateIndexMapWritable.put(new Text(key), new IntWritable(value));
		}
		hiddenStateIndexMapWritable.write(output);
		
		output.writeInt(featureIndexer.getysize());
		output.writeInt(featureIndexer.getmaxid());
	}

	static FeatureIndexer deserialize(DataInput input) throws IOException {
		Map<String, Integer> FeatureIndexMap = new HashMap<String, Integer>();
		Map<Integer, Integer> IndexFreqMap = new HashMap<Integer, Integer>();
		Map<String, Integer> HiddenStateIndexMap = new HashMap<String, Integer>();
		
		MapWritable featureIndexMapWritable = new MapWritable();
		featureIndexMapWritable.readFields(input);
		Set<Writable> keys=featureIndexMapWritable.keySet();//
		Iterator<Writable> it=keys.iterator();
		while(it.hasNext()){
			Text key=(Text)it.next();
			IntWritable value=(IntWritable)featureIndexMapWritable.get(key);
			FeatureIndexMap.put(key.toString(), new Integer(value.get()));
		}
		
		MapWritable indexFreqMapWritable = new MapWritable();
		indexFreqMapWritable.readFields(input);
		Set<Writable> keys_=indexFreqMapWritable.keySet();//
		Iterator<Writable> it_=keys_.iterator();
		while(it.hasNext()){
			IntWritable key=(IntWritable)it_.next();
			IntWritable value=(IntWritable)indexFreqMapWritable.get(key);
			IndexFreqMap.put(new Integer(key.get()), new Integer(value.get()));
		}
		
		MapWritable hiddenStateIndexMapWritable = new MapWritable();
		hiddenStateIndexMapWritable.readFields(input);
		Set<Writable> keys_2=hiddenStateIndexMapWritable.keySet();//
		Iterator<Writable> it_2=keys_2.iterator();
		while(it_2.hasNext()){
			Text key=(Text)it_2.next();
			IntWritable value=(IntWritable)hiddenStateIndexMapWritable.get(key);
			HiddenStateIndexMap.put(key.toString(), new Integer(value.get()));
		}
		
		int ysize=input.readInt();
		int maxid=input.readInt();
		return new FeatureIndexer(FeatureIndexMap,IndexFreqMap,HiddenStateIndexMap,ysize,maxid);
//		MapWritable featureIndex_Map = new MapWritable();
//		featureIndex_Map.readFields(input);
//		MapWritable indexFreq_Map = new MapWritable();
//		indexFreq_Map.readFields(input);
//		return new FeatureIndexer(featureIndex_Map,indexFreq_Map);
	}

}