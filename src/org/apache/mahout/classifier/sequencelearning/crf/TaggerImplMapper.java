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
import org.apache.hadoop.mapreduce.Mapper.Context;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class TaggerImplMapper extends Mapper<IntWritable, TaggerImplWritable, Text, VectorWritable>{
	Configuration conf;
	String CRFModel_Prior_PATH;
	
	CRFModel model;
	
	Vector alpha;
	Vector expected;
	Vector obj;
	Vector err;
	Vector zeroone;
	
	@Override  
	protected void setup(Context context) throws IOException,InterruptedException{
		conf=context.getConfiguration();
		CRFModel_Prior_PATH=conf.get("CRFModel_Prior_PATH");//?
		System.out.println("TaggerImplMapper:");
		System.out.println("CRFModel_Prior_PATH:"+CRFModel_Prior_PATH);
		
		Path serializerPath=new Path(CRFModel_Prior_PATH);
		FileSystem fs = FileSystem.get(URI.create(CRFModel_Prior_PATH), conf);
		DataInputStream inputstream= fs.open(serializerPath);
		model=LossyCRFModelSerializer.deserialize(inputstream);
		inputstream.close();
		
//		alpha=new DenseVector(model.maxid);
		alpha=model.alpha;
		expected=new DenseVector(model.maxid);
		obj=new DenseVector(1);
		err=new DenseVector(1);
		zeroone=new DenseVector(1);
		
		super.setup(context);
	}
	
	@Override
	protected void map(IntWritable key, TaggerImplWritable value, Context context)throws IOException,InterruptedException{
		
		TaggerImpl taggerImpl=value.getValue();
//		taggerImplDetails(taggerImpl);
		
		taggerImpl.alpha=alpha;//taggerImpl.alpha是空的
		taggerImpl.expected=expected;
		obj.set(0, obj.get(0)+taggerImpl.gradient());
		int error_num = taggerImpl.eval();
		err.set(0, err.get(0)+error_num);
		if(error_num!=0){
			zeroone.set(0, zeroone.get(0)+1);
		}
	}
	
	@Override
	protected void cleanup(Context context) throws IOException, InterruptedException {
		
		context.write(new Text("alpha"), new VectorWritable(alpha));
		context.write(new Text("expected"), new VectorWritable(expected));
		context.write(new Text("obj"), new VectorWritable(obj));
		context.write(new Text("err"), new VectorWritable(err));
		context.write(new Text("zeroone"), new VectorWritable(zeroone));
		
////		System.out.println("taggerImpl.gradient()=="+obj);
////		System.out.println("alpha.size()=="+alpha.size());//11160
////		System.out.println("alpha.get(0)=="+alpha.get(0));
////		System.out.println("expected.size()=="+expected.size());
////		System.out.println("expected.get(0)=="+expected.get(0));
////		System.out.println("expected.minValue()=="+expected.minValue());
//		/
		
		super.cleanup(context);
	}
	
	public void taggerImplDetails(TaggerImpl tagger){
		System.out.println("+++taggerImplDetails():");
		System.out.println("Greetings from TaggerImplMapper:");
		
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
		System.out.println("tagger.cost_factor="+tagger.cost_factor);
		
		System.out.println("此是全局的obj：taggerImpl.gradient()==="+obj);
		
		if(tagger.alpha.size()!=0){
			System.out.println("tagger.alpha.size()=="+tagger.alpha.size());//11160
			System.out.println("tagger.alpha.get(0)=="+tagger.alpha.get(0));
			System.out.println("tagger.alpha.minValue()=="+tagger.alpha.minValue());
		}
		if(tagger.expected.size()!=0){
			System.out.println("tagger.expected.size()=="+tagger.expected.size());
			System.out.println("tagger.expected.minValue()=="+tagger.expected.minValue());
			System.out.println("tagger.expected.get(0)=="+tagger.expected.get(0));
		}
		
		System.out.println("tagger.result="+tagger.result);
		System.out.println("tagger.nbest="+tagger.nbest);
		System.out.println("tagger.cost="+tagger.cost);
		System.out.println("tagger.Z="+tagger.Z);
		
//		System.out.println("tagger.nodeList.size()="+tagger.nodeList.size());
//		System.out.println("tagger.pathList.size()="+tagger.pathList.size());
		System.out.println("-----------------------------");
	}
}
