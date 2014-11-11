//[废弃]
package org.apache.mahout.classifier.sequencelearning.crf;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import java.io.IOException;
import java.net.URI;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.iterator.ClusterWritable;

public class FeatureIndexMapper extends Mapper<LongWritable, Text, Text, Text> {
	private static int xsize;
	private static String templatePath;
	private static String taggerPath;
	
	/*************************************************************************/
	private static HashMap<String,Integer> hm_feature=new HashMap<String,Integer>();
	private static ArrayList<String> al_feature=new ArrayList<String>();
	private static ArrayList<Integer> al_freq=new ArrayList<Integer>();
	private static HashMap<String,Integer> hm_hidden=new HashMap<String,Integer>();
	/*************************************************************************/
	
	SequenceFile.Writer writer = null;
	
	Text wordText = new Text();
	Text oneText=new Text("1");
	Text hiddenStateKey=new Text("#hiddenStateKey#");
	
	FeatureTemplate featureTemplate;
	FeatureExpander featureExpander;
	@Override  
	protected void setup(Context context){
		Configuration conf = context.getConfiguration();
		xsize=Integer.parseInt(conf.get("xsize"));
		templatePath=conf.get("Template_Path");
		taggerPath=conf.get("Tagger_Path")+"/part0000";
		
		
		try {
//			String randomUri="tagger1";
			FileSystem fs = FileSystem.get(URI.create(taggerPath), conf);
			Path path = new Path(taggerPath);
			Text key = new Text();
			TaggerImplWritable value = new TaggerImplWritable();
			writer = SequenceFile.createWriter(fs, conf, path,key.getClass(), value.getClass());
			
			featureTemplate=new FeatureTemplate(templatePath);
			featureExpander=new FeatureExpander(featureTemplate,xsize);
			super.setup(context);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
	}
  @Override
  protected void map(LongWritable key, Text value, Context context) throws IOException,
      InterruptedException {
	  String sentence=value.toString();
	  
	  TaggerImpl tagger=new TaggerImpl();
	  featureExpander.Expand(sentence,tagger);
	  int i=0;///
	  for(ArrayList<String> featureList : tagger.xStr){
//		  System.out.println("FeatureIndexMapper["+i+"]:"+featureList);///
		  for(String feature : featureList){
			  //\wordText.set(feature);
			  //\context.write(wordText, oneText);
			  /*************************************************************************/
//			  if(hm_feature.containsKey(feature)){
//				  Integer freq=hm_feature.get(feature)+1;
//				  hm_feature.put(feature, freq);
//			  }else{
//				  Integer freq=1;
//				  hm_feature.put(feature, freq);
//			  }
			  if(al_feature.contains(feature)){
				  int index=al_feature.indexOf(feature);
				  al_freq.set(index, al_freq.get(index)+1);
			  }else{
				  al_feature.add(feature);
				  al_freq.add(1);
			  }
			  /*************************************************************************/
		  }
		  i++;///
	  }
//	  System.out.println();///
	  for(String hiddenState : tagger.answerStr){
		  //\context.write(hiddenStateKey, new Text(hiddenState));
		  /*************************************************************************/
		  hm_hidden.put(hiddenState, null);
		  /*************************************************************************/
	  }
	  
	  /*写序列文件tagger*/
	  TaggerImplWritable taggerImplWritable=new TaggerImplWritable();
	  taggerImplWritable.setValue(tagger);
	  writer.append(new Text("taggerImplWritable"), taggerImplWritable);
	  
	  
//	  ArrayList<String> featureAL=featureExpander.Expand(sentence);
////	  System.out.println("featureAL:"+featureAL);
//	  Set<String> HiddenStateSet=featureExpander.getHiddenStateSet();
////	  System.out.println("PreTagSet:"+PreTagSet);
//	  
//	  for(int i=0;i<featureAL.size();i++){
//		  wordText.set(featureAL.get(i));
////		  System.out.println("key="+wordText.toString()+";value="+oneText.toString());
//		  context.write(wordText, oneText);
//	  }
//	  for(String hiddenState : HiddenStateSet){
////		  System.out.println("key="+preTagKey.toString()+";value="+preTag.toString());
//		  context.write(hiddenStateKey, new Text(hiddenState));
//	  }
	  
  }
  
  @Override
  protected void cleanup(Context context) throws IOException, InterruptedException {
	  writer.close();
	  /*************************************************************************/
	  
	  for(int i=0;i<al_feature.size();i++){
		  System.out.println("ID="+String.valueOf(i)+"	key="+al_feature.get(i)+"	value="+String.valueOf(al_freq.get(i)));
		  context.write(new Text(String.valueOf(i)),new Text(al_feature.get(i)+"@@"+String.valueOf(al_freq.get(i))));
	  }
	  
//	  int ID=0;
//	  Set<String> keys=hm_feature.keySet();
//	  Iterator<String> it=keys.iterator();
//	  while(it.hasNext()){
//		  String key=it.next();
//			Integer value=hm_feature.get(key);
//			System.out.println("ID="+ID+"	key="+key+"	value="+value);
//			context.write(new Text(String.valueOf(ID)),new Text(key+"@@"+String.valueOf(value)));
//			ID++;
////			System.out.println("key="+key+"	value="+value);
//		}
	  
//	  keys.clear();
	  Set<String> keys=hm_hidden.keySet();
	  Iterator<String> it=keys.iterator();
	  while(it.hasNext()){
		  String key=it.next();
			context.write(hiddenStateKey,new Text(key));
//			System.out.println("key="+key+"	value="+value);
		}
	  /*************************************************************************/
	  
    super.cleanup(context);
  }
  
  
  
}
