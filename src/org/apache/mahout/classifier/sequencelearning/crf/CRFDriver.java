package org.apache.mahout.classifier.sequencelearning.crf;

import static org.apache.mahout.clustering.topdown.PathDirectory.CLUSTERED_POINTS_DIRECTORY;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassificationDriver;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.ClassUtils;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.common.distance.SquaredEuclideanDistanceMeasure;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CRFDriver  extends AbstractJob{
	private static final Logger log = LoggerFactory.getLogger(CRFDriver.class);
	
	
	
	private static final String INITIAL_CRFMODEL_NAME = "CRFModel-Initial";
	
	public static void main(String[] args) throws Exception {
	    ToolRunner.run(new Configuration(), new CRFDriver(), args);
	}
	  
	@Override
	public int run(String[] args) throws Exception {
		return 0;////
	}

	//input="crfOutput/TaggerImpl";output="crfOutput"
	public static void run(Configuration conf, Path input, Path output, 		  
    		int maxitr,
    		double eta,
    		double C,
    		int shrinking_size,
    		String algorithm ) throws IOException, InterruptedException, ClassNotFoundException{
		
		Path CRFModelInitialPath=new Path(conf.get("CRFModelInitialPath"));
		System.out.println("CRFModelInitialPath:"+CRFModelInitialPath);
		
//		CrfLBFGS clbfgs=new CrfLBFGS();
//		LossyCrfLBFGSSerializer.serialize(clbfgs, output);
		
		
///		conf.set("InitialCRFModel_Path", output+"/"+ITERATION_DIR+"/"+INITIAL_CRFMODEL_NAME);
		
////		String CRFModel_Initial_Path=output.toString()+"/"+INITIAL_CRFMODEL_NAME;
////		conf.set("CRFModel_Initial_Path", CRFModel_Initial_Path);//指定初始化crf模型的生成路径crfOutput/CRFModel-Initial
////		buildInitialCRFModel(conf);
////		System.out.println("+++showCRFModel():");
////		showCRFModel(CRFModel_Initial_Path);
//		\
		
		System.out.println("-------------------new CRFIterator().iterateMR(0)---------------------------------------------");
//		Path priorCRFModelPath = new Path(output, INITIAL_CRFMODEL_NAME);//priorCRFModelPath=CRFModel_Initial_Path
//		new CRFIterator().iterateMR_Once(conf, input, CRFModelInitialPath, output, maxitr,eta);
		new CRFIterator().iterateMR(conf, input, CRFModelInitialPath, output, maxitr,eta);
		
	}
	
	
	private static void buildInitialCRFModel(Configuration conf) throws IOException{
		  log.info("build Initial CRFModel");
		  Path CRFModelInitialPath = new Path(conf.get("CRFModel_Initial_Path"));
		  
		  CRFModel model=new CRFModel(Integer.parseInt(conf.get("maxid")),"");
//		  model.setFeatureIndexerSerializerURI(conf.get("FeatureIndexer_Serializer_URI"));
		  FileSystem fs=FileSystem.get(URI.create(CRFModelInitialPath.toString()), conf);
		  DataOutputStream outstream=fs.create(CRFModelInitialPath);
		  LossyCRFModelSerializer.serialize(model, outstream);
		  outstream.close();
		  
	  }
	private static CRFModel showCRFModel(String uri) throws IOException{
		  Configuration conf = new Configuration();
			
			Path  serializerPath=new Path(uri);
			try {
				FileSystem fs = FileSystem.get(URI.create(uri), conf);
				DataInputStream inputstream=fs.open(serializerPath);
				CRFModel model=LossyCRFModelSerializer.deserialize(inputstream);
				
				System.out.println("model.maxid="+model.maxid);
//			    System.out.println("model.ysize="+model.ysize);
//			    System.out.println("model.featureIndexerSerializerURI="+model.featureIndexerSerializerURI);
			    System.out.println("model.alpha.size()="+model.alpha.size());
			    System.out.println("model.alpha.minValue()="+model.alpha.minValue());
			    System.out.println("model.alpha.get(0)="+model.alpha.get(0));
			    System.out.println("model.expected.size()="+model.expected.size());
			    System.out.println("model.expected.minValue()="+model.expected.minValue());
			    System.out.println("model.expected.get(0)="+model.expected.get(0));
				
				inputstream.close();
				return model;
			} catch (IOException e1) {
				// TODO Auto-generated catch block
				e1.printStackTrace();
			}
			return null;
	  }	
	
	
	
}
