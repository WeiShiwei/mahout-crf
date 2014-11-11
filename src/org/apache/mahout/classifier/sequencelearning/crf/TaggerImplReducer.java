package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Reducer.Context;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import riso.numerical.*;

public class TaggerImplReducer extends Reducer<Text,VectorWritable,Text,DoubleWritable>{
	Configuration conf;
	String CRFModel_Prior_PATH;
	String CRFModel_Dest_PATH;
	String CrfLBFGS_Prior_PATH;//?
	String CrfLBFGS_Dest_PATH;//?
	
	CRFModel model;
	CrfLBFGS clbfgs;
	
	Vector alpha;
	Vector expected;
	Vector obj;
	Vector err;
	Vector zeroone;
	
	@Override  
	protected void setup(Context context) throws IOException,InterruptedException{
		System.out.println("BuildTaggerImplReducer:");
		
		conf=context.getConfiguration();
		CRFModel_Prior_PATH=conf.get("CRFModel_Prior_PATH");
		CRFModel_Dest_PATH=conf.get("CRFModel_Dest_PATH");
		System.out.println("CRFModel_Dest_PATH:"+CRFModel_Dest_PATH);
		
		//CRFModel的反序列化
		Path serializerPath=new Path(CRFModel_Prior_PATH);
		FileSystem fs = FileSystem.get(URI.create(CRFModel_Prior_PATH), conf);
		DataInputStream inputstream= fs.open(serializerPath);
		model=LossyCRFModelSerializer.deserialize(inputstream);
		inputstream.close();
		
		//CrfLBFGS的反序列化
		if(conf.get("CrfLBFGS_Prior_PATH").equals("NA")){
			clbfgs=new CrfLBFGS();//\
		}else{
			CrfLBFGS_Prior_PATH=conf.get("CrfLBFGS_Prior_PATH");
			Path serializerPath1=new Path(CrfLBFGS_Prior_PATH);
			FileSystem fs1 = FileSystem.get(URI.create(CrfLBFGS_Prior_PATH), conf);
			DataInputStream inputstream1= fs1.open(serializerPath1);
			clbfgs=LossyCrfLBFGSSerializer.deserialize(inputstream1);
			inputstream1.close();
		}
		System.out.println("CrfLBFGS_Prior_PATH:"+CrfLBFGS_Prior_PATH);
		CrfLBFGS_Dest_PATH=conf.get("CrfLBFGS_Dest_PATH");
		System.out.println("CrfLBFGS_Dest_PATH:"+CrfLBFGS_Dest_PATH);
		

		alpha=new DenseVector(model.maxid);
		expected=new DenseVector(model.maxid);
		obj=new DenseVector(1);
		err=new DenseVector(1);
		zeroone=new DenseVector(1);		
		
		super.setup(context);
	}
	
	@Override
	protected void reduce(Text key, Iterable<VectorWritable> values, Context context){
		
		if(key.toString().equals("obj")){
			Iterator<VectorWritable> iter = values.iterator();
			while (iter.hasNext()) {
				obj=obj.plus(iter.next().get());
			}
		}else if(key.toString().equals("expected")){
			Iterator<VectorWritable> iter = values.iterator();
			while (iter.hasNext()) {
				expected=expected.plus(iter.next().get());
			}
		}else if(key.toString().equals("alpha")){
			Iterator<VectorWritable> iter = values.iterator();
			while (iter.hasNext()) {
				alpha=alpha.plus(iter.next().get());
			}
		}else if(key.toString().equals("err")){
			Iterator<VectorWritable> iter = values.iterator();
			while (iter.hasNext()) {
				err=err.plus(iter.next().get());
			}
		}else if(key.toString().equals("zeroone")){
			Iterator<VectorWritable> iter = values.iterator();
			while (iter.hasNext()) {
				zeroone=zeroone.plus(iter.next().get());
			}
		}		
	}
	
	@Override
	protected void cleanup(Context context) throws IOException, InterruptedException {
		int n=model.maxid;//参数个数
		double x [ ]= new double [ n ];//参数向量
		double f ; //目标函数值
		double g [ ] = new double [ n ];//梯度向量
		int iflag[] = new int[1];iflag[0]=0;
		
		for(int k=0;k<alpha.size();k++){//目标函数值和期望向量用罚函数更新
			obj.set(0, obj.get(0)+(alpha.get(k)*alpha.get(k)/(2.0*1.0)));
			expected.set(k,expected.get(k)+alpha.get(k)/1.0);
		}
		/*赋予x,g,f*/
		for(int i=0;i<alpha.size();i++){
			x[i]=alpha.get(i);
			g[i]=expected.get(i);
		}
		f=obj.get(0);		
		
		clbfgs.optimize(n, x, f, g, iflag);//x(参数向量)和f(目标函数值)被更新
		
		/**更新alpha和obj*/
		for(int k=0;k<x.length;k++){
			alpha.set(k, x[k]);
		}
		obj.set(0, f);
		
//		/*lbfgs*/
//		int ndim=model.maxid;int msave=7;
//		int nwork=ndim * ( 2 * msave + 1 ) + 2 * msave ;
//		double x [ ] , g [ ] , diag [ ] , w [ ];
//		x = new double [ ndim ];///
//		g = new double [ ndim ];///
//		diag = new double [ ndim ];
//		w = new double [ nwork ];
//		
//		double f, eps, xtol;
//		int iprint [ ] , iflag[] = new int[1], icall, n, m;
//		iprint = new int [ 2 ];
//		boolean diagco;
//		
//		n=model.maxid;//{x0,x1...xn}共有n+1个x
//		m=5;
//		iprint [ 1 -1] = 1;
//		iprint [ 2 -1] = 0;
//		diagco= false;
//		eps= 1.0e-5;
//		xtol= 1.0e-16;
//		icall=0;
//		iflag[0]=0;
//		
//		for(int k=0;k<alpha.size();k++){//罚函数
////			obj+=(alpha.get(k)*alpha.get(k)/(2.0*1.0));
//			obj.set(0, obj.get(0)+(alpha.get(k)*alpha.get(k)/(2.0*1.0)));
//			expected.set(k,expected.get(k)+alpha.get(k)/1.0);
//		}
//		for(int i=0;i<alpha.size();i++){
//			x[i]=alpha.get(i);
//			g[i]=expected.get(i);
//		}
//		f=obj.get(0);
//		
//		try{
//			/**************************************A调试**********************************************/
//			
//			System.out.println("------TaggerImplReducer------");
//			System.out.println("alpha详细信息：");
//			for(int hh=0;hh<alpha.size();hh++){
//				System.out.println("alpha["+hh+"]	"+alpha.get(hh));
//			}
//			System.out.println();
//			System.out.println("obj="+obj.get(0));
////			
//			System.out.println("expected详细信息：");
//			for(int hh=0;hh<expected.size();hh++){
//				System.out.println(hh+"	"+expected.get(hh)+"	");
//			}
//			/**************************************B**********************************************/
//			LBFGS.lbfgs ( n , m , x , f , g , diagco , diag , iprint , eps , xtol , iflag );
//		}
//		catch (LBFGS.ExceptionWithIflag e){
//			System.err.println( "Sdrive: lbfgs failed.\n"+e );
//			return;
//		}
//		//暂时还是不要用BFGS的iflag了
//		//if(iflag[0] == 0){}
//		
//		/**更新alpha*/
//		for(int k=0;k<x.length;k++){
//			alpha.set(k, x[k]);
//		}
//		obj.set(0, f);
		
		context.write(new Text("old_obj"), new DoubleWritable(model.obj));//context.write()
		
		model.alpha=alpha;
		model.expected=expected;
		model.obj=obj.get(0);
		model.err=(int)err.get(0);
		model.zeroone=(int)zeroone.get(0);
		
		//序列化存储CRFModel
		FileSystem fs=FileSystem.get(URI.create(CRFModel_Dest_PATH), conf);
		DataOutputStream outstream=fs.create(new Path(CRFModel_Dest_PATH));
		LossyCRFModelSerializer.serialize(model, outstream);
		outstream.close();
		//序列化存储LossyCrfLBFGSSerializer
		FileSystem fs1=FileSystem.get(URI.create(CrfLBFGS_Dest_PATH), conf);
		DataOutputStream outstream1=fs1.create(new Path(CrfLBFGS_Dest_PATH));
		LossyCrfLBFGSSerializer.serialize(clbfgs, outstream1);//\修改点
		outstream1.close();
		
		
		context.write(new Text("obj"), new DoubleWritable(model.obj));//context.write()
		
		super.cleanup(context);
	}
	
}
