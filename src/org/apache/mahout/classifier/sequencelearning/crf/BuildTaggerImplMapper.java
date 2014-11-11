package org.apache.mahout.classifier.sequencelearning.crf;

import java.util.HashSet;
import java.util.Set;

import java.io.IOException;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;

public class BuildTaggerImplMapper extends Mapper<LongWritable, Text, IntWritable, TaggerImplWritable> {
	private static Set<String> HiddenStateSet=new HashSet<String>();

	private static int xsize;
	private static String templatePath;
	
	FeatureTemplate featureTemplate;
	FeatureExpander featureExpander;
	@Override  
	protected void setup(Context context) throws IOException,InterruptedException{
		
		Configuration conf = context.getConfiguration();
		xsize=Integer.parseInt(conf.get("xsize"));
		templatePath=conf.get("TemplatePath");
		
		featureTemplate=new FeatureTemplate(templatePath);
		featureExpander=new FeatureExpander(featureTemplate,xsize);
		super.setup(context);
	}
	@Override
	protected void map(LongWritable key, Text value, Context context) throws IOException,
	      InterruptedException {
		System.out.println("Begin BuildTaggerImplMapper.map()>:"+System.currentTimeMillis());
		String sentence=value.toString();
		TaggerImpl tagger=new TaggerImpl();
		featureExpander.Expand(sentence,tagger);
		
		for(String hiddenState : tagger.answerStr){
			HiddenStateSet.add(hiddenState);
		}
		
		TaggerImplWritable taggerImplWritable=new TaggerImplWritable();
		taggerImplWritable.setValue(tagger);
		context.write(new IntWritable(1), taggerImplWritable);
		System.out.println("End BuildTaggerImplMapper.map()<:"+System.currentTimeMillis());
	}
	
	@Override
	protected void cleanup(Context context) throws IOException, InterruptedException {
//		Configuration conf = context.getConfiguration();
//		String hiddenStateList="";
//		for(String hiddenState : HiddenStateSet){
//			hiddenStateList+=hiddenState+"@@";
//		}
//		conf.set("HiddenStateList", hiddenStateList);
//		System.out.println("BuildTaggerMapper()::hiddenStateList="+hiddenStateList);
		super.cleanup(context);
	}
}
