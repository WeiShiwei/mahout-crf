package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.ReflectionUtils;
import org.apache.hadoop.util.ToolRunner;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.common.AbstractJob;
import org.apache.mahout.common.HadoopUtil;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public final class Job extends AbstractJob {
  
  private static final Logger log = LoggerFactory.getLogger(Job.class);///【slf4j】
  
  //private static final String DIRECTORY_CONTAINING_TrainCorpus_INPUT = "data";  
  
  private Job() {}
  
  public static void main(String[] args) throws Exception {
    if (args.length > 0) {
      log.info("Running with only user-supplied arguments");
      ToolRunner.run(new Configuration(), new Job(), args);
    } else {///CRF一切从这里开始
      log.info("Running with default arguments");
      Path output = new Path("crfOutput");///默认的输出目录
      Configuration conf = new Configuration();
      HadoopUtil.delete(conf, output);
      
      run(conf, new Path("crfInput"),new Path("template"), output, 3,10000,0.0001,1.0,20,"CRF");//另一个输入文件夹crfInput2xsize
    }
  }
  
  @Override
  public int run(String[] args) throws Exception {///处理args参数的run()方法我先不管
    return 0;
  }
  
  /**
   * Run the CRF
   * 
   * @param conf
   *          the Configuration to use
   * @param input
   *          the String denoting the input directory path
   * @param output
   *          the String denoting the output directory path
   * @param templfile
   *          模板文件
   * @param trainfile
   *          存储训练语料的文件夹
   * @param xsize
   * 		  训练语料在横向上维度（2的代表是分词，3的代表是浅层分析）(注意CRF++中的xsize从0开始的，这里xsize是从1开始数的自然数，取值2或3)
   * @param ysize
   * 		  训练语料隐藏（预测）状态的数量（应该说程序应该自动区分并计算这个参数值，但是我觉得与xsize和ysize类似，这两个参数事先很容易确定，但是对宏观的影响非常大，这个权利交给用户）       
   * 		  xsize代表我的任务，ysize代表标注语料库
   * @param modelfile***模型文件（目标，这个参数现在不需要，要他默认生成就行了）
   * @param boolean textmodelfile***输出的模型是文本的还是二进制的build also text model file for debugging（默认值是0）          
   * @param maxitr
   *          set INT for max iterations in LBFGS routine(default 10k)（默认值10000）
   * @param freq***use features that occuer no less than INT(default 1)(默认值是1)
   * @param eta
   *   		  set FLOAT for termination criterion(default 0.0001)（默认值0.0001）
   * @param C
   *          set FLOAT for cost parameter(default 1.0)（默认值是1.0） 
   * @param shrinking_size
   *          set INT for number of iterations variable needs to be optimal before considered for shrinking. (默认值 20)
   * @param algorithm
   *          select training algorithm（MIRA|CRF）
   */
  public static void run(Configuration conf, Path input,Path templatePath, Path output, 		  
		int xsize,
		int maxitr,
		double eta,
		double C,
		int shrinking_size,
		String algorithm ) throws Exception {
	
	//输入目录input="crfInput"，输出目录output="output"
////	  Path corpusPath=new Path(output,"Corpus");
////    Path TaggerImplPath = new Path(output, "TaggerImpl");
////    Path CRFModelInitialPath = new Path(output, "CRFModel-Initial");
//	  \
    log.info("Preparing Input");
    conf.set("TemplatePath", templatePath.toString());
    
    /*
     * 数据的预处理
     * InputDriver:
     * input="crfInput";
     * output="crfOutput"
     * templatePath="crfInput/template"
     */
    /***************************************************************/
    System.out.println("Before InputDriver.buildTaggerImpl()函数:"+System.currentTimeMillis());
    InputDriver.buildTaggerImpl(conf,input,output,xsize);
    System.out.println("After InputDriver.buildTaggerImpl()函数:"+System.currentTimeMillis());
    /***************************************************************/
////    InputDriver.buildTaggerImpl(conf,new Path(corpusPath,"tagger"),TaggerImplPath);
////    InputDriver.buildFeatureIndex(conf,input,corpusPath,xsize);
////    InputDriver.buildFeatureIndexerSerializer(conf);
////    InputDriver.buildTaggerImpl(conf,new Path(conf.get("Tagger_Path")),TaggerImplPath);
//    \
    
///    log.info("Running CRF");
///    Path TaggerImplPath = new Path(conf.get("TaggerImplPath"));
    
///    System.out.println("TaggerImplPath:"+TaggerImplPath);
///    CRFDriver.run(conf, TaggerImplPath, output, maxitr,eta,C,shrinking_size,"L-BFGS");
    

  
  }
  
  
  
}
