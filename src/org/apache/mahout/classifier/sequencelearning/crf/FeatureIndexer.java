//[废弃]
package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class FeatureIndexer {
	private int ysize=0;//隐藏状态集合的大小，在IndexingHiddenState函数中更新设置
	private int maxid=0;//特征函数的个数，在IndexingFeatureIndex函数中更新设置
	//注意：如果Writable类型作为Map容器的类型会出错的，遍历的时候全部是最后一个元素，但我不知道为什么
	private Map<String, Integer> FeatureIndexMap = new HashMap<String, Integer>();
	private Map<Integer, Integer> IndexFreqMap = new HashMap<Integer, Integer>();
	private Map<String, Integer> HiddenStateIndexMap = new HashMap<String, Integer>();
	
	/**
	 * 构造函数
	 *  @param uri
	 *  特征索引数据的路径，crfOutput/FeatureIndex/data
	 *  @param YSIZE
	 *  YSIZE是预测标记集合(隐藏状态集合)大小，即PreTagIndexMap的集合大小
	 * @throws IOException 
	 */
	public FeatureIndexer(String hiddenStateUri,String featureIndexUri) throws IOException{
		IndexingHiddenState(hiddenStateUri);//两者的顺序不能乱，此方法除了填充HiddenStateIndexMap还更新ysize
		IndexingFeatureIndex(featureIndexUri);
	}
	/**
	 * 构造函数（用于序列化）
	 * @param featureIndexHashMap
	 * @param indexFreqHashMap
	 * @param HiddenStateIndexMap
	 * @param ysize
	 * @param maxid
	 */
	public FeatureIndexer(Map<String, Integer> featureIndexHashMap,Map<Integer, Integer> indexFreqHashMap,Map<String, Integer> hiddenStateIndexMap,
			int ysize,int maxid){
		FeatureIndexMap=featureIndexHashMap;
		IndexFreqMap=indexFreqHashMap;
		HiddenStateIndexMap=hiddenStateIndexMap;
		this.ysize=ysize;
		this.maxid=maxid;
	}
	
	/**
	 * 
	 * @param hiddenStateUri
	 * 隐藏状态及索引文件路径
	 * @throws IOException
	 */
	private void IndexingHiddenState(String hiddenStateUri) throws IOException{
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(URI.create(hiddenStateUri), conf);
		Path pathHS = new Path(hiddenStateUri);
		SequenceFile.Reader reader = null;
		try {
			reader = new SequenceFile.Reader(fs, pathHS, conf);
			Text key = (Text)ReflectionUtils.newInstance(reader.getKeyClass(), conf);//ReflectionUtils
			IntWritable value = (IntWritable)ReflectionUtils.newInstance(reader.getValueClass(), conf);
			String hiddenState;
			int index;
			
			long position = reader.getPosition();
			while (reader.next(key, value)) {
//				String syncSeen = reader.syncSeen() ? "*" : "";
//				System.out.printf("[%s%s]\t%s\t%s\n", position, syncSeen, key, value);
				
				hiddenState=key.toString();
				index=value.get();
				HiddenStateIndexMap.put(hiddenState, index);
				ysize++;
				
				position = reader.getPosition();
			}
		}finally {
			IOUtils.closeStream(reader);
		}
	}
	
	/**
	 * @param featureFreqUri
	 * 特征和频数数据文件路径
	 * @throws IOException
	 */
	private void IndexingFeatureIndex(String featureIndexUri) throws IOException{
		//读序列文件
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(URI.create(featureIndexUri), conf);
		Path path = new Path(featureIndexUri);

		SequenceFile.Reader reader = null;
		try {
			reader = new SequenceFile.Reader(fs, path, conf);
			Text key = (Text)ReflectionUtils.newInstance(reader.getKeyClass(), conf);//ReflectionUtils
			IntWritable value = (IntWritable)ReflectionUtils.newInstance(reader.getValueClass(), conf);
			
			String feature=new String();
			Integer index=new Integer(0);
			Integer freq=new Integer(0);
			
			long position = reader.getPosition();
			while (reader.next(key, value)) {
//				String syncSeen = reader.syncSeen() ? "*" : "";
//				System.out.printf("[%s%s]\t%s\t%s\n", position, syncSeen, key, value);
				
				/*
				 * 判断操作：
				 * key是feature，要判断feature是不是U或者是B，如果是U则featureIndex+=ysize;如果是B则featureIndex+=ysize*ysize;
				 */
				//理论上MapReduce的键一定是唯一的，所以if(!FeatureIndexMap.containsKey(key))分支一定会进去的
				//效率上也没有损失，只是Map遍历的顺序（即键的顺序）不保证等于index的顺序（但是仔细推论的话这句话其实也不对），要了解mapreduce的输出!
				//暂时由MapReduce输出的序列文件来保证正确性
				if(!FeatureIndexMap.containsKey(key.toString())){
					//<feature,index>写入FeatureIndexMap; <index,freq>写入IndexFreqIndexMap
					feature=key.toString();
					index=new Integer(maxid);
					freq=new Integer(value.get());
					//System.out.println("feature:"+feature+"	index:"+index+"	freq:"+freq);
					
					FeatureIndexMap.put(feature,index);
					////System.out.println("FeatureIndexMap.put(key="+feature+", value="+index+")=========="+FeatureIndexMap.get(feature));
					IndexFreqMap.put(index, freq);
					////System.out.println("IndexFreqMap.put(key="+index+", value="+freq+")=========="+IndexFreqMap.get(index));
					
					if(feature.startsWith("U")){
						maxid+=ysize;
					}else{
						maxid+=ysize*ysize;
						//System.out.println("FeatureIndexMap.put(key="+feature+", value="+index+")=========="+FeatureIndexMap.get(feature));
						//System.out.println("FeatureIndexMap.put(key="+feature+", value="+index+")=========="+FeatureIndexMap.get(feature));
					}
					
				}
				
				position = reader.getPosition(); // beginning of next record
			}
		}finally {
			IOUtils.closeStream(reader);
		}
//		System.out.println("FeatureIndexMap.size()="+FeatureIndexMap.size());
//		System.out.println("IndexFreqMap.size()="+IndexFreqMap.size());
		
	}
	
	/**
	 * @param featureAL
	 * 例如：featureAL={U00:_B-2, U01:_B-1, U02:Confidence...,
						U00:_B-1, U01:Confidence, U02:in...,
						...}
	 * @return Vector
	 * 返回索引矩阵;不同的mapper同时调用该函数
	 *
	 */
	public void Register(TaggerImpl tagger){
		for(ArrayList<String> featurelist : tagger.xStr){
			ArrayList<Integer> fvector=new ArrayList<Integer>();
			for(String feature:featurelist){
				fvector.add(FeatureIndexMap.get(feature));
			}
			tagger.x.add(fvector);
		}
		
		for(String hiddenstate : tagger.answerStr){
			tagger.answer.add(HiddenStateIndexMap.get(hiddenstate));
		}
		tagger.xsize=tagger.answerStr.size();
		tagger.ysize=ysize;
	}
	public void Register(ArrayList<ArrayList<String> > xS,ArrayList<String> answerS,ArrayList<ArrayList<Integer> > x,ArrayList<Integer> answer){
		//xS，answerS是外部定义的变量，然后经过FeatureExpander.ExpendUB(xS，answerS)处理，得到实际值
		for(ArrayList<String> featurelist : xS){
			ArrayList<Integer> fvector=new ArrayList<Integer>();
			for(String feature:featurelist){
				fvector.add(FeatureIndexMap.get(feature));
			}
			x.add(fvector);
		}
		
		for(String hiddenstate : answerS){
			answer.add(HiddenStateIndexMap.get(hiddenstate));
		}
	}
//	public Vector FeatureIndexRegister(ArrayList<String> featureAL) throws IOException{
//		if(featureAL.isEmpty()){
//			return null;
//		}
//		
//		Vector featureIndexVec= new DenseVector(featureAL.size());
//		double[] values=new double[featureAL.size()];
//		for(int i=0;i<featureAL.size();i++){
//			String key=featureAL.get(i);
//			values[i]=FeatureIndexMap.get(key);
//		}
//		featureIndexVec.assign(values);
//		
//		return featureIndexVec;
//	}
	
	/*
	 * showFeatureIndexMap()
	 * showIndexFreqMap()
	 * showPreTagIndexMap()
	 */
	public void showFeatureIndexMap(){
		Set<String> keys=FeatureIndexMap.keySet();
		Iterator<String> it=keys.iterator();
		while(it.hasNext()){
			String key=it.next();
			Integer value=FeatureIndexMap.get(key);
			System.out.println("key="+key+"	value="+value);
		}
	}
	public void showIndexFreqMap(){
		Set<Integer> keys=IndexFreqMap.keySet();
		Iterator<Integer> it=keys.iterator();
		while(it.hasNext()){
			Integer key=it.next();
			Integer value=IndexFreqMap.get(key);
			System.out.println("key="+key+"	value="+value);
		}
	}
	public void showHiddenStateIndexMap(){
		Set<String> keys=HiddenStateIndexMap.keySet();
		Iterator<String> it=keys.iterator();
		while(it.hasNext()){
			String key=it.next();
			Integer value=HiddenStateIndexMap.get(key);
			System.out.println("key="+key+"	value="+value);
		}
	}
	
	/*
	 * get函数
	 */
	public Map<String, Integer> getFeatureIndexMap(){
		return FeatureIndexMap;
	}
	public Map<Integer, Integer> getIndexFreqMap(){
		return IndexFreqMap;
	}
	public Map<String, Integer> getHiddenStateIndexMap(){
		return HiddenStateIndexMap;
	}
	public int getmaxid(){
		return maxid;
	}
	public int getysize(){
		return ysize;
	}
	
	
	
}
	/**
	public void Indexing(String uri,int ysize) throws IOException{
		int k=0;
		//读序列文件
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(URI.create(uri), conf);
		Path path = new Path(uri);

		SequenceFile.Reader reader = null;
		try {
			reader = new SequenceFile.Reader(fs, path, conf);
			Text key = (Text)ReflectionUtils.newInstance(reader.getKeyClass(), conf);//ReflectionUtils
			IntWritable value = (IntWritable)ReflectionUtils.newInstance(reader.getValueClass(), conf);
			
			Text feature=new Text();
			IntWritable index=new IntWritable();
			IntWritable freq=new IntWritable();
			
			
			long position = reader.getPosition();
			while (reader.next(key, value)) {
				String syncSeen = reader.syncSeen() ? "*" : "";
				System.out.printf("[%s%s]\t%s\t%s\n", position, syncSeen, key, value);
				
				//理论上MapReduce的键一定是唯一的，所以if(!FeatureIndex_Map.containsKey(key))分支一定会进去的
				//效率上也没有损失，只是Map遍历的顺序（即键的顺序）不保证等于index的顺序（但是仔细推论的话这句话其实也不对），要了解mapreduce的输出!
				//暂时由MapReduce输出的序列文件来保证正确性
				if(!FeatureIndex_Map.containsKey(key)){
					k++;
					//<feature,index>写入FeatureIndex_Map; <index,freq>写入IndexFreqIndex_Map
					feature=key;
					index=new IntWritable(featureIndex);
					freq=value;
					System.out.println("feature:"+feature.toString()+"	index:"+featureIndex+"	freq:"+freq.toString());
					
					FeatureIndex_Map.put(feature,index);
					System.out.println("FeatureIndex_Map.put(key="+feature.toString()+", value="+index.toString()+")=========="+FeatureIndex_Map.get(feature));
					
					IndexFreq_Map.put(index, freq);
					System.out.println("IndexFreq_Map.put(key="+index.toString()+", value="+freq.toString()+")=========="+IndexFreq_Map.get(index));
					
					featureIndex+=ysize;
				}
				
				position = reader.getPosition(); // beginning of next record
			}
		}finally {
			IOUtils.closeStream(reader);
		}

		System.out.println("FeatureIndex_Map.size()="+FeatureIndex_Map.size());
		System.out.println("IndexFreq_Map.size()="+IndexFreq_Map.size());
		System.out.println("k="+k);
	}
	
	public void showFeatureIndex_Map(){
		
				MapWritable FeatureIndexMap=FeatureIndex_Map;
				Set<Writable> keys=FeatureIndexMap.keySet();
				Iterator<Writable> it=keys.iterator();
				while(it.hasNext()){
					Text key=(Text)it.next();
					IntWritable value=(IntWritable)FeatureIndexMap.get(key);
					System.out.println("key="+key.toString()+"	value="+value.toString());
				}
	}
	
	public FeatureIndexer(MapWritable featureIndex_Map,MapWritable indexFreq_Map){
		FeatureIndex_Map=featureIndex_Map;
		IndexFreq_Map=indexFreq_Map;
	}
	public MapWritable getFeatureIndexMap(){
		return FeatureIndex_Map;
	}
	public MapWritable getIndexFreqMap(){
		return IndexFreq_Map;
	}
	**/
	
	//--------------------------------------------------------------------------------------------------------//
	/**
	 static int featureIndex=0;
	//注意：如果Writable类型作为Map容器的类型会出错的，遍历的时候全部是最后一个元素，但我不知道
	Map<Text, IntWritable> FeatureIndexMap = new HashMap<Text, IntWritable>();
	Map<IntWritable, IntWritable> IndexFreqMap = new HashMap<IntWritable, IntWritable>();
//	MapWritable FeatureIndex_Map = new MapWritable();
//	MapWritable IndexFreq_Map = new MapWritable();
	
	public FeatureIndexer(){}
	
	public void IndexingHashMap(String uri,int ysize) throws IOException{
		int k=0;
		//读序列文件
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(URI.create(uri), conf);
		Path path = new Path(uri);

		SequenceFile.Reader reader = null;
		try {
			reader = new SequenceFile.Reader(fs, path, conf);
			Text key = (Text)ReflectionUtils.newInstance(reader.getKeyClass(), conf);//ReflectionUtils
			IntWritable value = (IntWritable)ReflectionUtils.newInstance(reader.getValueClass(), conf);
			
			Text feature=new Text();
			IntWritable index=new IntWritable();
			IntWritable freq=new IntWritable();
			
			
			long position = reader.getPosition();
			while (reader.next(key, value)) {
				String syncSeen = reader.syncSeen() ? "*" : "";
				System.out.printf("[%s%s]\t%s\t%s\n", position, syncSeen, key, value);
				
				//理论上MapReduce的键一定是唯一的，所以if(!FeatureIndexMap.containsKey(key))分支一定会进去的
				//效率上也没有损失，只是Map遍历的顺序（即键的顺序）不保证等于index的顺序（但是仔细推论的话这句话其实也不对），要了解mapreduce的输出!
				//暂时由MapReduce输出的序列文件来保证正确性
				if(!FeatureIndexMap.containsKey(key)){
					k++;
					//<feature,index>写入FeatureIndex_Map; <index,freq>写入IndexFreqIndex_Map
					feature=key;
					index=new IntWritable(featureIndex);
					freq=value;
					System.out.println("feature:"+feature+"	index:"+index+"	freq:"+freq);
					
					FeatureIndexMap.put(feature,index);
					System.out.println("FeatureIndexMap.put(key="+feature+", value="+index+")=========="+FeatureIndexMap.get(feature));
					
					IndexFreqMap.put(index, freq);
					System.out.println("IndexFreqMap.put(key="+index+", value="+freq+")=========="+IndexFreqMap.get(index));
					
					featureIndex+=ysize;
				}
				
				position = reader.getPosition(); // beginning of next record
			}
		}finally {
			IOUtils.closeStream(reader);
		}
		System.out.println("FeatureIndexMap.size()="+FeatureIndexMap.size());
		System.out.println("IndexFreqMap.size()="+IndexFreqMap.size());
		System.out.println("k="+k);
	}
	
	public void showFeatureIndexMap(){
		Set<Text> keys=FeatureIndexMap.keySet();
		Iterator<Text> it=keys.iterator();
		while(it.hasNext()){
			Text key=it.next();
			IntWritable value=FeatureIndexMap.get(key);
			System.out.println("key="+key+"	value="+value);
		}
	}
	
	public FeatureIndexer(Map<Text, IntWritable> featureIndexHashMap,Map<IntWritable, IntWritable> indexFreqHashMap){
		FeatureIndexMap=featureIndexHashMap;
		IndexFreqMap=indexFreqHashMap;
	}
	public Map<Text, IntWritable> getFeatureIndexMap(){
		return FeatureIndexMap;
	}
	public Map<IntWritable, IntWritable> getIndexFreqMap(){
		return IndexFreqMap;
	}
	 */
	
	

