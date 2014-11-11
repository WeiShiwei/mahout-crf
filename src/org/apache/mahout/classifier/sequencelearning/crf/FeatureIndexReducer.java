//[废弃]
package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.IOException;
import java.net.URI;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Set;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.SequenceFile;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class FeatureIndexReducer extends Reducer<Text,Text,Text,IntWritable> {
	private static String HiddenStateUri;
	
	SequenceFile.Writer writer = null;
	
	Text hiddenStateKey=new Text("#hiddenStateKey#");
	IntWritable zero = new IntWritable(0);
	
	@Override  
	protected void setup(Context context){
		Configuration conf = context.getConfiguration();
		HiddenStateUri=conf.get("HiddenState_URI");
		try {
			FileSystem fs = FileSystem.get(URI.create(HiddenStateUri), conf);
			Path path = new Path(HiddenStateUri);
			Text key = new Text();
			IntWritable value = new IntWritable();
			writer = SequenceFile.createWriter(fs, conf, path,key.getClass(), value.getClass());
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		
		try {
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
  protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException,
      InterruptedException {
	  Iterator<Text> iter = values.iterator();
	  
	  if(!key.equals(hiddenStateKey)){
		  //\int Frequency=0;
		  //\while (iter.hasNext()) {
			  //\Frequency+= Integer.parseInt(iter.next().toString());
		  //\}
		  //\context.write(key, new IntWritable(Frequency));
		  /***********************************************************************************/
		  iter.hasNext();
		  String feature_freq=iter.next().toString();
		  System.out.println("feature_freq:"+feature_freq);
		  String[] ff=feature_freq.split("@@");
		  context.write(new Text(ff[0]), new IntWritable(Integer.parseInt(ff[1])));
		  /***********************************************************************************/
	  }else{
		  //因为HiddenStateKey是唯一的，所以只在一个reducer里面进入else分支，没有多reducer的顾虑
		  Set<String> HiddenStateSet=new HashSet<String>();
		  while (iter.hasNext()) {
			  HiddenStateSet.add(iter.next().toString());
		  }
		  int i=0;
		  for(String hiddenState : HiddenStateSet){
			  writer.append(new Text(hiddenState), new IntWritable(i));
			  i++;
		  }
		  IOUtils.closeStream(writer);
	  }
	
  }
   
  
}
