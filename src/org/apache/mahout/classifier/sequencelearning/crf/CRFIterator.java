
package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.IOException;
import java.net.URI;
import java.util.Iterator;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.ClusteringPolicy;
import org.apache.mahout.common.iterator.sequencefile.PathFilters;
import org.apache.mahout.common.iterator.sequencefile.PathType;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileDirValueIterable;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileValueIterator;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import com.google.common.io.Closeables;

/**
 * This is a clustering iterator which works with a set of Vector data and a prior ClusterClassifier which has been
 * initialized with a set of models. Its implementation is algorithm-neutral and works for any iterative clustering
 * algorithm (currently k-means, fuzzy-k-means and Dirichlet) that processes all the input vectors in each iteration.
 * The cluster classifier is configured with a ClusteringPolicy to select the desired clustering algorithm.
 */
public class CRFIterator {
  
  public static final String PRIOR_PATH_KEY = "org.apache.mahout.clustering.prior.path";
  private static final String ITERATION_Prefix = "CRFModel-";  
  private static int converge=0;
//  double old_obj;
//  double obj;
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of iterations
   * 
   * @param policy
   *          the ClusteringPolicy to use
   * @param data
   *          a {@code List<Vector>} of input vectors
   * @param classifier
   *          a prior ClusterClassifier
   * @param numIterations
   *          the int number of iterations to perform
   * 
   * @return the posterior ClusterClassifier
   */
  public ClusterClassifier iterate(Iterable<Vector> data, ClusterClassifier classifier, int numIterations) {
    ClusteringPolicy policy = classifier.getPolicy();
    for (int iteration = 1; iteration <= numIterations; iteration++) {
      for (Vector vector : data) {
        // update the policy based upon the prior
        policy.update(classifier);
        // classification yields probabilities
        Vector probabilities = classifier.classify(vector);
        // policy selects weights for models given those probabilities
        Vector weights = policy.select(probabilities);
        // training causes all models to observe data
        for (Iterator<Vector.Element> it = weights.iterateNonZero(); it.hasNext();) {
          int index = it.next().index();
          classifier.train(index, vector, weights.get(index));
        }
      }
      // compute the posterior models
      classifier.close();
    }
    return classifier;
  }
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of iterations using a sequential
   * implementation
   * 
   * @param conf
   *          the Configuration
   * @param inPath
   *          a Path to input VectorWritables
   * @param priorPath
   *          a Path to the prior classifier
   * @param outPath
   *          a Path of output directory
   * @param numIterations
   *          the int number of iterations to perform
   * 
   * @throws IOException
   */
  public void iterateSeq(Configuration conf, Path inPath, Path priorPath, Path outPath, int numIterations)
      throws IOException {
    ClusterClassifier classifier = new ClusterClassifier();
    classifier.readFromSeqFiles(conf, priorPath);
    Path clustersOut = null;
    int iteration = 1;
    while (iteration <= numIterations) {
      for (VectorWritable vw : new SequenceFileDirValueIterable<VectorWritable>(inPath, PathType.LIST,
          PathFilters.logsCRCFilter(), conf)) {
        Vector vector = vw.get();
        // classification yields probabilities
        Vector probabilities = classifier.classify(vector);
        // policy selects weights for models given those probabilities
        Vector weights = classifier.getPolicy().select(probabilities);
        // training causes all models to observe data
        for (Iterator<Vector.Element> it = weights.iterateNonZero(); it.hasNext();) {
          int index = it.next().index();
          classifier.train(index, vector, weights.get(index));
        }
      }
      // compute the posterior models
      classifier.close();
      // update the policy
      classifier.getPolicy().update(classifier);
      // output the classifier
      clustersOut = new Path(outPath, Cluster.CLUSTERS_DIR + iteration);
      classifier.writeToSeqFiles(clustersOut);
      FileSystem fs = FileSystem.get(outPath.toUri(), conf);
      iteration++;
//      if (isConverged(clustersOut, conf, fs)) {
      if (true){
        break;
      }
    }
    Path finalClustersIn = new Path(outPath, Cluster.CLUSTERS_DIR + (iteration - 1) + Cluster.FINAL_ITERATION_SUFFIX);
    FileSystem.get(clustersOut.toUri(), conf).rename(clustersOut, finalClustersIn);
  }
  
  /**
   * Iterate over data using a prior-trained ClusterClassifier, for a number of iterations using a mapreduce
   * implementation
   * 
   * @param conf
   *          the Configuration
   * @param inPath
   *          a Path to input VectorWritables
   * @param priorPath
   *          a Path to the prior classifier
   * @param outPath
   *          a Path of output directory
   * @param numIterations
   *          the int number of iterations to perform
 * @throws IOException 
 * @throws ClassNotFoundException 
 * @throws InterruptedException 
   */
  public void iterateMR_Once(Configuration conf, Path inPath, Path priorPath, Path outPath, int numIterations,double eta) throws IOException, InterruptedException, ClassNotFoundException{
	  Path clustersOut = null;
	  conf.set("PRIOR_PATH", priorPath.toString());//
	  
	  int iteration=1;
      
	  String jobName = "Cluster Iterator running iteration " + iteration + " over priorPath: " + priorPath;
      System.out.println(jobName);
      Job job = new Job(conf, jobName);
      
      job.setMapOutputKeyClass(Text.class);
      job.setMapOutputValueClass(VectorWritable.class);
      job.setOutputKeyClass(Text.class);
      job.setOutputValueClass(VectorWritable.class);
      
      job.setInputFormatClass(SequenceFileInputFormat.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
      job.setMapperClass(TaggerImplMapper.class);
      job.setReducerClass(TaggerImplReducer.class);
      job.setNumReduceTasks(1);
      
      FileInputFormat.addInputPath(job, inPath);
      clustersOut = new Path(outPath, ITERATION_Prefix + iteration);
      priorPath = clustersOut;
      FileOutputFormat.setOutputPath(job, clustersOut);///
      
      job.setJarByClass(CRFIterator.class);
      if (!job.waitForCompletion(true)) {
        throw new InterruptedException("Cluster Iteration " + iteration + " failed processing " + priorPath);
      }
      
//      old_obj=obj;
//      obj=Double.parseDouble(conf.get("obj"));
      
  }
  /**
   * MapReduce迭代算法
   * @param conf
   * @param inPath="crfOutput/TaggerImpl"
   * @param CRFModelPriorPath=conf.get("CRFModelInitialPath")="crfOutput/CRFModelInitial"
   * @param outPath="crfInput/template"
   * @param numIterations（默认值10000）
   * @param eta
   * set FLOAT for termination criterion(default 0.0001)（默认值0.0001）
   * 
   * @throws IOException
   * @throws InterruptedException
   * @throws ClassNotFoundException
   */
	public void iterateMR(Configuration conf, Path inPath,Path CRFModelPriorPath, Path outPath, int numIterations, double eta)
			throws IOException, InterruptedException, ClassNotFoundException {
		
		//CRFModelPriorPath="/crfOutput/CRFModelInitial"
		Path CrfLBFGSPriorPath=null;
		Path clustersOut = null;
		
		int iteration = 1;
		// while (iteration <= numIterations) {
		while (iteration <= 15) {
			conf.set("CRFModel_Prior_PATH", CRFModelPriorPath.toString());//CRFModel_Prior_PATH
			if(CrfLBFGSPriorPath==null){
				conf.set("CrfLBFGS_Prior_PATH", "NA");
			}else{
				conf.set("CrfLBFGS_Prior_PATH", CrfLBFGSPriorPath.toString());
			}
			
			///ITERATION_Prefix = "CRFModel-"
			clustersOut = new Path(outPath, ITERATION_Prefix + iteration);
			
			Path CRFModelDestPath = new Path(clustersOut, "model");
			conf.set("CRFModel_Dest_PATH", CRFModelDestPath.toString());//CRFModel_Dest_PATH
			
			Path crfLBFGSDestPATH = new Path(clustersOut, "lbfgs");
			conf.set("CrfLBFGS_Dest_PATH", crfLBFGSDestPATH.toString());//CrfLBFGS_Dest_PATH

			
			String jobName = "Cluster Iterator running iteration " + iteration
					+ " over priorPath: " + CRFModelPriorPath;
			System.out.println(jobName);
			Job job = new Job(conf, jobName);

			job.setMapOutputKeyClass(Text.class);
			job.setMapOutputValueClass(VectorWritable.class);
			job.setOutputKeyClass(Text.class);
			job.setOutputValueClass(DoubleWritable.class);

			job.setInputFormatClass(SequenceFileInputFormat.class);
			job.setOutputFormatClass(SequenceFileOutputFormat.class);
			job.setMapperClass(TaggerImplMapper.class);
			job.setReducerClass(TaggerImplReducer.class);
			job.setNumReduceTasks(1);

			FileInputFormat.addInputPath(job, inPath);
			CRFModelPriorPath = CRFModelDestPath;//CRFModelPriorPath
			CrfLBFGSPriorPath = crfLBFGSDestPATH;//CrfLBFGSPriorPath
			FileOutputFormat.setOutputPath(job, clustersOut);

			job.setJarByClass(CRFIterator.class);
			if (!job.waitForCompletion(true)) {
				throw new InterruptedException("Cluster Iteration " + iteration
						+ " failed processing " + CRFModelPriorPath);
			}
			FileSystem fs = FileSystem.get(outPath.toUri(), conf);
			
			iteration++;

//			Double old_obj = new Double(0.0);
//			Double obj = new Double(0.0);
//			getObj(conf, clustersOut, old_obj, obj);
//
//			if (isConverged(clustersOut, conf, fs, iteration, numIterations,eta, old_obj, obj)) {
//				break;
//			}

		}
		System.out.println("迭代结束");
//		Path finalClustersIn = new Path(outPath, ITERATION_Prefix+ (iteration - 1));
//		FileSystem.get(clustersOut.toUri(), conf).rename(clustersOut,finalClustersIn);
	}
  
  private void getObj(Configuration conf,Path objPath,Double old_obj,Double obj) throws IOException{
		System.out.println("old_obj=" + old_obj + " obj=" + obj);
		/****************************************************************/
		
		Path pathUri=new Path(objPath,"part-r-00000");
////		String Uri = "hdfs://localhost:9000/user/weishiwei/CRFModel-1/part-r-00000";
//		\

		FileSystem fs = FileSystem.get(pathUri.toUri(), conf);
		SequenceFile.Reader reader = null;
		try {
			reader = new SequenceFile.Reader(fs, pathUri, conf);
			Text key = (Text) ReflectionUtils.newInstance(
					reader.getKeyClass(), conf);// ReflectionUtils
			DoubleWritable value = (DoubleWritable) ReflectionUtils
					.newInstance(reader.getValueClass(), conf);

			long position = reader.getPosition();
			while (reader.next(key, value)) {
				if(key.toString().equals("old_obj")){
					old_obj=value.get();
				}
				if(key.toString().equals("obj")){
					obj=value.get();
				}
				position = reader.getPosition();
			}
		} finally {
			IOUtils.closeStream(reader);
		}
		
		/****************************************************************/
		System.out.println("old_obj=" + old_obj + " obj=" + obj);
  } 
  
  /**
   * 
   * @param filePath
   * @param conf
   * @param fs
   * @param itr
   * @param maxitr
   * @param eta
   * @param old_obj
   * @param obj
   * @return
   * @throws IOException
   */
  private boolean isConverged(Path filePath, Configuration conf, FileSystem fs,
		  int itr,int maxitr,double eta,
		  double old_obj,double obj) throws IOException {
	  //如果filePath指定的crf模型中有old_obj和obj就好了
	  double diff = (itr == 1 ? 1.0 : Math.abs(old_obj - obj)/old_obj);///diff是"相对误差限"
	  
	  if (diff < eta) {///eta=9.99e-005
	      converge++;///如果相对误差限diff小于eta，则converge++
	  }else{
	      converge = 0;
	  }
	  
	  ///迭代次数itr大于迭代次数限制，或者converge=3；退出
	  if (itr > maxitr || converge == 3) {
		  return true;  // 3 is ad-hoc[ad-hoc:特定的、自定义?]
	  }
	  
	  return false;
    
  }
  
  
}
