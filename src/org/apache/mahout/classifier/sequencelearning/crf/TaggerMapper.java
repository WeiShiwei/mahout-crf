//[废弃]
package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.DataInputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.mahout.math.Vector;

//public class TaggerImplMapper  extends Mapper<LongWritable, Text, IntWritable, TaggerImplWritable>{
public class TaggerMapper  extends Mapper<Text, TaggerImplWritable, IntWritable, TaggerImplWritable>{
	String pahtToFeatureIndexSerializer;
	FeatureIndexer featureIndexer;
	
//	private static int xsize;
//	private static int ysize;
	static ArrayList<ArrayList<String> > xS;
	static ArrayList<String> answerS;
//	private static String pahtToFeatureIndexSerializer;
//	private static FeatureExpander featureExpander;
//	private static FeatureIndexer featureIndexer;
	
	@Override  
	protected void setup(Context context){
		Configuration conf = context.getConfiguration();
		pahtToFeatureIndexSerializer=conf.get("FeatureIndexer_Serializer_URI");
		
		Path  serializerPath=new Path(pahtToFeatureIndexSerializer);
		FileSystem fs;
		try {
			fs = FileSystem.get(URI.create(pahtToFeatureIndexSerializer), conf);
			DataInputStream inputstream=fs.open(serializerPath);
			featureIndexer=FeatureIndexSerializer.deserialize(inputstream);
			inputstream.close();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
//		xsize=Integer.parseInt(conf.get("xsize"));
//		ysize=Integer.parseInt(conf.get("ysize"));
//		try {
//			featureExpander=new FeatureExpander(new FeatureTemplate("123"),xsize);
//		} catch (IOException e1) {
//			e1.printStackTrace();
//		}
		
//		String uri1="hdfs://localhost:9000/user/weishiwei/crfOutput/FeatureIndex/HiddenStateIndex";//隐藏状态索引文件
//		String uri2="hdfs://localhost:9000/user/weishiwei/crfOutput/FeatureIndex/data/part-r-00000";//特征索引数据文件
//		featureIndexer=new FeatureIndexer(uri1, uri2);
		
		
//		pahtToFeatureIndexSerializer=conf.get("pahtToFeatureIndexSerializer");
		try {
			super.setup(context);
		} catch (IOException e) {
			e.printStackTrace();
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}
	
	  @Override
	  protected void map(Text key, TaggerImplWritable value, Context context) throws IOException, InterruptedException {
		  TaggerImpl tagger=value.getValue();
		  featureIndexer.Register(tagger);
		  TaggerImplWritable taggerImplWritable=new TaggerImplWritable();//构建TaggerImplWritable对象
		  taggerImplWritable.setValue(tagger);
		  context.write(new IntWritable(1), taggerImplWritable);
		  
		  
		  /*验证*/
		  
		  System.out.println("Greetings from TaggerMapper:");
		  int i=1;
			for(ArrayList<String> flist : tagger.xStr){
				  System.out.println("xStr"+i+":"+flist);
				  i++;
			  }
			 System.out.println("tagger.answerStr="+tagger.answerStr);
			 
			 int j=1;
				for(ArrayList<Integer> flist : tagger.x){
					  System.out.println("x"+j+":"+flist);
					  j++;
				  }
				System.out.println("tagger.answer="+tagger.answer);
				
			 System.out.println("tagger.xsize="+tagger.xsize);
			 System.out.println("tagger.ysize="+tagger.ysize);
			 System.out.println("-----------------------------");
		  
//		  Configuration conf=new Configuration();
//		  Path  serializerPath=new Path(pahtToFeatureIndexSerializer);
//		  FileSystem fs=FileSystem.get(URI.create(pahtToFeatureIndexSerializer), conf);
//		  DataInputStream inputstream=fs.open(serializerPath);
//		  FeatureIndexer featureIndexer=FeatureIndexSerializer.deserialize(inputstream);
//		  inputstream.close();
		  
//		  featureExpander.ExpendUB(sentence, xS, answerS);
//		  TaggerImpl tagger=new TaggerImpl();
//		  featureIndexer.Register(xS, answerS, tagger.x, tagger.answer);
//		  
//		  TaggerImplWritable taggerImplWritable=new TaggerImplWritable();//构建TaggerImplWritable对象
//		  taggerImplWritable.setValue(tagger);
//
//		  
//		  context.write(new IntWritable(1), taggerImplWritable);
		  
		  
//		  mapper中特征注册且返回索引，这里并不去计算频数，因为所有的mapper都在争抢这个“函数”资源，让这个函数执行的时间越短越好
//		  考虑到所有mapper都结束之后才开始reducer，那么map函数中，可以这样设计：
//		  （1）不要用context写taggerImplWritable，用一个普通函数写到指定文件
//		  （2）context写特征串和频数，考虑patitioner优化，reduce中统计特征串频数，然后在更新索引文件
//		  总结：好像不太合适，这样所有mapper不仅抢“函数”，还抢索引文件;否定之～20130410
	  }

	  
}


//ArrayList<String> featureAL=featureExpander.Expand(sentence);//生成特征
////反序列化
////pahtToFeatureIndexSerializer="hdfs://localhost:9000/user/weishiwei/crfOutput/FeatureIndex/featureIndexSerializer";
//System.out.println("serializerPath:"+serializerPath.toString());
//
//
//Vector featureIndexVec=featureIndexerD.FeatureIndexRegister(featureAL);//特征注册，返回索引
//
//TaggerImplWritable taggerImplWritable=new TaggerImplWritable();//构建TaggerImplWritable对象
//int ysize=featureExpander.getysize();//Ysize是这个TaggerImplMapper处理句子的token的数量
//taggerImplWritable.setValue(new TaggerImpl(featureIndexVec,ysize));//构建TaggerImpl对象