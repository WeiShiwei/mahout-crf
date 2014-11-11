package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.DataOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Iterator;
import java.util.TreeSet;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.mapreduce.Reducer;

public class BuildTaggerImplReducer  extends Reducer<IntWritable,TaggerImplWritable,IntWritable,TaggerImplWritable>{
	private static TreeSet<String> HiddenStateSet=new TreeSet<String>();
	private static String CRFModelInitialPath;
	TaggerImpl tagger;
	FeatureIndex featureIndex=new FeatureIndex();

	@Override  
	protected void setup(Context context) throws IOException,InterruptedException {
		Configuration conf = context.getConfiguration();
		CRFModelInitialPath=conf.get("CRFModelInitialPath");
		
//		String hiddenStateList=conf.get("HiddenStateList");
//		String hiddenStateList="B-NP@@B-PP@@B-VP@@I-NP@@O";
		String hiddenStateList="B-ADJP@@B-ADVP@@B-NP@@B-PP@@B-PRT@@B-SBAR@@B-VP@@I-ADJP@@I-ADVP@@I-NP@@I-PP@@I-SBAR@@I-VP@@O";
		System.out.println("BuildTaggerReducer()::hiddenStateList="+hiddenStateList);
		String[] hiddenStateArray=hiddenStateList.split("@@");
		for(int i=0;i<hiddenStateArray.length;i++){
			HiddenStateSet.add(hiddenStateArray[i]);
		}
//		System.out.println("遍历HiddenStateSet");
//		for(String hiddenState : HiddenStateSet){
//			System.out.println("hiddenState="+hiddenState);
//		}
		featureIndex.IndexingHStateIndex(HiddenStateSet);/*20131101问题：HiddenStateSet来源有问题啊*/		
		super.setup(context);
		
	}
	
	@Override
	protected void reduce(IntWritable key, Iterable<TaggerImplWritable> values, Context context) throws IOException,InterruptedException {
		System.out.println("Begin BuildTaggerImplMapper.reduce()>:"+System.currentTimeMillis());
		
		Iterator<TaggerImplWritable> iter = values.iterator();
		while (iter.hasNext()) {
			tagger=iter.next().getValue();
			featureIndex.IndexingFeatureIndex(tagger);
			featureIndex.Register(tagger);
		}
		TaggerImplWritable taggerImplWritable=new TaggerImplWritable();
		taggerImplWritable.setValue(tagger);
		context.write(new IntWritable(1), taggerImplWritable);
		System.out.println("End BuildTaggerImplMapper.reduce()>:"+System.currentTimeMillis());
	}
	
	@Override
	protected void cleanup(Context context) throws IOException, InterruptedException {
		//序列化存储featureIndex.2
		
		int maxid=featureIndex.getMaxID();
		//序列化存储CRFModel
		Configuration conf = context.getConfiguration();
		CRFModel model=new CRFModel(maxid,"featureIndexURI");
		FileSystem fs=FileSystem.get(URI.create(CRFModelInitialPath), conf);
		DataOutputStream outstream=fs.create(new Path(CRFModelInitialPath));
		LossyCRFModelSerializer.serialize(model, outstream);
		outstream.close();
		
		super.cleanup(context);
	}
}
