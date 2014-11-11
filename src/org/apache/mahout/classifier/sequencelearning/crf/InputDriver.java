/**
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;

import org.apache.commons.cli2.CommandLine;
import org.apache.commons.cli2.Group;
import org.apache.commons.cli2.Option;
import org.apache.commons.cli2.OptionException;
import org.apache.commons.cli2.builder.ArgumentBuilder;
import org.apache.commons.cli2.builder.DefaultOptionBuilder;
import org.apache.commons.cli2.builder.GroupBuilder;
import org.apache.commons.cli2.commandline.Parser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapred.MapFileOutputFormat;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.conversion.InputMapper;
import org.apache.mahout.clustering.iterator.CIMapper;
import org.apache.mahout.clustering.iterator.CIReducer;
import org.apache.mahout.clustering.iterator.ClusterIterator;
import org.apache.mahout.clustering.iterator.ClusterWritable;
import org.apache.mahout.common.CommandLineUtil;
import org.apache.mahout.common.IntPairWritable;
import org.apache.mahout.common.commandline.DefaultOptionCreator;
import org.apache.mahout.math.VectorWritable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public final class InputDriver {

  private static final Logger log = LoggerFactory.getLogger(InputDriver.class);

  //  private static final String INITIAL_CRFModel_NAME = "Model-0";
//  static String crfOutput_Path;
//  private static String Tagger_Path ;
//  private static String Corpus_Path;
//  private static String HiddenState_URI ;
//  private static String FeatureIndexer_Serializer_URI ;
//public static void defaultPathConfigurationCenter(Path crfOutput){
//  Path Corpus_Path=new Path(crfOutput,"Corpus");
//  Path Feature_Path=new Path(crfOutput,"feature");
//  Path HiddenStateIndex_URI=new Path(crfOutput,"hiddenStateIndex");
//  Path Tagger_Path=new Path(crfOutput,"tagger");
//  Path FeatureIndexer_Serializer_URI=new Path(Corpus_Path,"featureIndexerSerializer");
//  Path TaggerImpl_Path = new Path(crfOutput, "TaggerImpl");
//}
  
  private InputDriver() {  }
  /**
  public static void main(String[] args) throws IOException, InterruptedException, ClassNotFoundException {
    DefaultOptionBuilder obuilder = new DefaultOptionBuilder();
    ArgumentBuilder abuilder = new ArgumentBuilder();
    GroupBuilder gbuilder = new GroupBuilder();
    
    Option inputOpt = DefaultOptionCreator.inputOption().withRequired(false).create();
    Option outputOpt = DefaultOptionCreator.outputOption().withRequired(false).create();
    Option vectorOpt = obuilder.withLongName("vector").withRequired(false).withArgument(
      abuilder.withName("v").withMinimum(1).withMaximum(1).create()).withDescription(
      "The vector implementation to use.").withShortName("v").create();
    
    Option helpOpt = DefaultOptionCreator.helpOption();
    
    Group group = gbuilder.withName("Options").withOption(inputOpt).withOption(outputOpt).withOption(
      vectorOpt).withOption(helpOpt).create();

    try {
      Parser parser = new Parser();
      parser.setGroup(group);
      CommandLine cmdLine = parser.parse(args);
      if (cmdLine.hasOption(helpOpt)) {
        CommandLineUtil.printHelp(group);
        return;
      }
      
      Path input = new Path(cmdLine.getValue(inputOpt, "testdata").toString());
      Path output = new Path(cmdLine.getValue(outputOpt, "output").toString());
      String vectorClassName = cmdLine.getValue(vectorOpt,
         "org.apache.mahout.math.RandomAccessSparseVector").toString();
      runJob(input, output, vectorClassName);
    } catch (OptionException e) {
      log.error("Exception parsing command line: ", e);
      CommandLineUtil.printHelp(group);
    }
  }**/
  
  public static void runJob(Path input, Path output, String vectorClassName)
    throws IOException, InterruptedException, ClassNotFoundException {
//    Configuration conf = new Configuration();
//    conf.set("vector.implementation.class.name", vectorClassName);
//    Job job = new Job(conf, "Input Driver running over input: " + input);
//
//    job.setOutputKeyClass(Text.class);
//    //job.setOutputValueClass(VectorWritable.class);//
//    job.setOutputValueClass(Text.class);
//    job.setOutputFormatClass(SequenceFileOutputFormat.class);//
//    
//    job.setMapperClass(FeatureIndexMapper.class);   
//    job.setNumReduceTasks(0);
//    job.setJarByClass(InputDriver.class);
//    
//    FileInputFormat.addInputPath(job, input);
//    FileOutputFormat.setOutputPath(job, output);
//    
//    boolean succeeded = job.waitForCompletion(true);
//    if (!succeeded) {
//      throw new IllegalStateException("Job failed!");
//    }
  }
  

  
  public static void buildTaggerImpl(Configuration conf,Path input,Path output,int xsize) throws IOException, InterruptedException, ClassNotFoundException{
	  conf.set("xsize", String.valueOf(xsize));
	  
	  Path CRFModelInitialPath = new Path(output, "CRFModelInitial");
	  conf.set("CRFModelInitialPath", CRFModelInitialPath.toString());
	  
	  Path TaggerImplPath = new Path(output, "TaggerImpl");
	  conf.set("TaggerImplPath", TaggerImplPath.toString());
	  
	  String jobName = "buildTagger";
	  System.out.println(jobName);
	  
	  Job job = new Job(conf, jobName);
	  job.setMapOutputKeyClass(IntWritable.class);
      job.setMapOutputValueClass(TaggerImplWritable.class);
	  job.setOutputKeyClass(IntWritable.class);
	  job.setOutputValueClass(TaggerImplWritable.class);
	  job.setOutputFormatClass(SequenceFileOutputFormat.class);
	  
	  job.setInputFormatClass(TextInputFormat.class);
      job.setOutputFormatClass(SequenceFileOutputFormat.class);
	  
	  job.setMapperClass(BuildTaggerImplMapper.class);
	  job.setReducerClass(BuildTaggerImplReducer.class);
	  job.setNumReduceTasks(1);
	  job.setJarByClass(InputDriver.class);
	  
	  FileInputFormat.addInputPath(job, input);
	  FileOutputFormat.setOutputPath(job, TaggerImplPath);
	  
	  job.setJarByClass(InputDriver.class);
	  if (!job.waitForCompletion(true)) {
	     throw new InterruptedException("");
	  }
  }
  
//  public static void buildTaggerImpl(Configuration conf,Path input,Path output) throws IOException, InterruptedException, ClassNotFoundException{
//	  String jobName = "buildTaggerImpl";
//	  System.out.println();
//	  System.out.println(jobName);
//	  Job job = new Job(conf, jobName);
//	  
//	  job.setMapOutputKeyClass(IntWritable.class);
//	  job.setMapOutputValueClass(TaggerImplWritable.class);
//	  job.setOutputKeyClass(IntWritable.class);
//	  job.setOutputValueClass(TaggerImplWritable.class);
//	  
//	  job.setInputFormatClass(SequenceFileInputFormat.class);
//	  job.setOutputFormatClass(SequenceFileOutputFormat.class);
//	  job.setNumReduceTasks(1);
//	  
////	  job.setMapperClass(BuildTaggerImplMapper.class);
////	  job.setReducerClass(BuildTaggerImplReducer.class);
//	  
//	  FileInputFormat.addInputPath(job, input);
//	  FileOutputFormat.setOutputPath(job, output);
//	  
//	  job.setJarByClass(InputDriver.class);
//	  if (!job.waitForCompletion(true)) {
//	     throw new InterruptedException("");
//	  }
//  }
  
//  public static void buildFeatureIndex(Configuration conf,Path input,Path output,int xsize) throws IOException, InterruptedException, ClassNotFoundException{
////	  Configuration conf = new Configuration();
//	  Path featureDatePath=new Path(output,"feature");
////	  Path featurePath=new Path(output,"feature");
////	  Path hiddenStatePath=new Path(output,"hiddenStateIndex");
////	  Path taggerPath=new Path(output,"tagger");
////	  Path output=new Path(conf.get("Corpus_PATH"));
////	  Corpus_Path=output.toString();
//	  conf.set("Corpus_Path", output.toString());
//	  
//	  conf.set("xsize", String.valueOf(xsize));
//	  conf.set("FeatureDate_Path", featureDatePath.toString());
//	  conf.set("HiddenState_URI", output.toString()+"/hiddenStateIndex");
//	  conf.set("Tagger_Path", output.toString()+"/tagger");
//	  
//	  
//	  String jobName = "buildFeatureIndex";
//	  System.out.println(jobName);
//	  Job job = new Job(conf, jobName);
//	  job.setMapOutputKeyClass(Text.class);
//	  job.setMapOutputValueClass(Text.class);
//	  job.setOutputKeyClass(Text.class);
//	  job.setOutputValueClass(IntWritable.class);//
//	  
//	  job.setInputFormatClass(TextInputFormat.class);
//	  job.setOutputFormatClass(TextOutputFormat.class);
//	  //job.setOutputFormatClass(SequenceFileOutputFormat.class);
//	  job.setNumReduceTasks(1);
////	  可以尝试序列文件的压缩
////	  SequenceFileOutputFormat.setCompressOutput(job, true);
////	  SequenceFileOutputFormat.setOutputCompressorClass(job, GzipCodec.class);
////	  SequenceFileOutputFormat.setOutputCompressionType(job,CompressionType.BLOCK);
//	  
//	  job.setMapperClass(FeatureIndexMapper.class);
//	  job.setReducerClass(FeatureIndexReducer.class);
//	  
//	  FileInputFormat.addInputPath(job, input);
//	  
//	  FileOutputFormat.setOutputPath(job, featureDatePath);
//	  
//	  job.setJarByClass(InputDriver.class);
//	  if (!job.waitForCompletion(true)) {
//	     throw new InterruptedException("");
//	  }
//	  
//  } 
  
//  public static void buildFeatureIndexerSerializer(Configuration conf) throws IOException{
//	  System.out.println("buildFeatureIndexerSerializer");
//	  
//	  Path featureIndexerSerializerURI=new Path(new Path(conf.get("Corpus_Path")),"featureIndexerSerializer");
//	  conf.set("FeatureIndexer_Serializer_URI", featureIndexerSerializerURI.toString());
//	  String hiddenState_URI=conf.get("HiddenState_URI");
//	  
//	  String uri2="hdfs://localhost:9000/user/weishiwei/crfOutput/Corpus/feature/part-r-00000";//特征索引数据文件
//	  FeatureIndexer featureIndexer=new FeatureIndexer(hiddenState_URI, uri2);//隐藏状态索引文件,特征索引数据文件
////	  featureIndexer.showHiddenStateIndexMap();
////	  featureIndexer.showFeatureIndexMap();
////	  featureIndexer.showIndexFreqMap();
//	  FileSystem fs=FileSystem.get(URI.create(featureIndexerSerializerURI.toString()), conf);
//	  DataOutputStream outstream=fs.create(featureIndexerSerializerURI);//路径应该是crfOutput/FeatureIndex/featureIndexSerializer
//	  FeatureIndexSerializer.serialize(featureIndexer, outstream);
//	  outstream.close();
////	  System.out.println("featureIndexer.getysize()"+featureIndexer.getysize());//8
////	  System.out.println("featureIndexer.getmaxid()"+featureIndexer.getmaxid());//11160
//	  conf.set("maxid", String.valueOf(featureIndexer.getmaxid()));
//	  conf.set("ysize", String.valueOf(featureIndexer.getysize()));
//	
//  }
  
  
//  public static void buildTaggerImpl(Configuration conf,Path input,Path output) throws IOException, InterruptedException, ClassNotFoundException{//(Path input, Path pahtToFeatureIndexSerializer,Path output)
//
//	  String jobName = "buildTaggerImpl";
//	  System.out.println();
//	  System.out.println(jobName);
//	  Job job = new Job(conf, jobName);
//	  
//	  job.setMapOutputKeyClass(IntWritable.class);
//      job.setMapOutputValueClass(ClusterWritable.class);
//	  job.setOutputKeyClass(IntWritable.class);
//	  job.setOutputValueClass(TaggerImplWritable.class);
////	  job.setOutputFormatClass(SequenceFileOutputFormat.class);
//	  
//	  job.setInputFormatClass(SequenceFileInputFormat.class);
//      job.setOutputFormatClass(SequenceFileOutputFormat.class);
//	    
//	  job.setMapperClass(TaggerMapper.class);   
//	  job.setNumReduceTasks(0);
//	  job.setJarByClass(InputDriver.class);
//	    
//	  FileInputFormat.addInputPath(job, input);
//	  FileOutputFormat.setOutputPath(job, output);
//	  
//	  job.setJarByClass(InputDriver.class);
//	  if (!job.waitForCompletion(true)) {
//	     throw new InterruptedException("");
//	  }
//  }
  
  
  
//public static FeatureIndexer showFeatureIndexSerializer(String uri) throws IOException{
//	
//	Configuration conf = new Configuration();
//	
//	Path  serializerPath=new Path(uri);
//	try {
//		FileSystem fs = FileSystem.get(URI.create(uri), conf);
//		DataInputStream inputstream=fs.open(serializerPath);
//		FeatureIndexer featureIndexer=FeatureIndexSerializer.deserialize(inputstream);
//		
//		System.out.println("featureIndexer.getmaxid():"+featureIndexer.getmaxid());
//		System.out.println("featureIndexer.getysize():"+featureIndexer.getysize());
//		
//		
//		inputstream.close();
//		return featureIndexer;
//	} catch (IOException e1) {
//		// TODO Auto-generated catch block
//		e1.printStackTrace();
//	}
//	return null;
//	
//}
//  public static void taggerseqfile(String taggeruri,FeatureIndexer featureIndexer) throws IOException{
//	  Configuration conf = new Configuration();
//	  FileSystem fs1 = FileSystem.get(URI.create(taggeruri), conf);
//		Path pathHS = new Path(taggeruri);
//		SequenceFile.Reader reader = null;
//		try {
//			reader = new SequenceFile.Reader(fs1, pathHS, conf);
//			Text key = (Text)ReflectionUtils.newInstance(reader.getKeyClass(), conf);//ReflectionUtils
//			TaggerImplWritable value = (TaggerImplWritable)ReflectionUtils.newInstance(reader.getValueClass(), conf);
//			
//			long position = reader.getPosition();
//			while (reader.next(key, value)) {
//				String syncSeen = reader.syncSeen() ? "*" : "";
//				//System.out.printf("[%s%s]\t%s\t%s\n", position, syncSeen, key, value);
//				
//				TaggerImpl tagger=value.getValue();
//				featureIndexer.Register(tagger);////featureIndexer
//				
//				int i=1;
//				for(ArrayList<String> flist : tagger.xStr){
//					  System.out.println("xStr"+i+":"+flist);
//					  i++;
//				  }
//				 System.out.println("tagger.answerStr="+tagger.answerStr);
//				 
//				 int j=1;
//					for(ArrayList<Integer> flist : tagger.x){
//						  System.out.println("x"+j+":"+flist);
//						  j++;
//					  }
//					System.out.println("tagger.answer="+tagger.answer);
//					
//				 System.out.println("tagger.xsize="+tagger.xsize);
//				 System.out.println("tagger.ysize="+tagger.ysize);
////				 System.out.println("tagger.answer="+tagger.answer);
//				 
//				 System.out.println("-----------------------------");
//				
//				position = reader.getPosition();
//			}
//		}finally {
//			IOUtils.closeStream(reader);
//		}
//  }
  
  
  ///////////////////////////////////////////////////////////////
}
