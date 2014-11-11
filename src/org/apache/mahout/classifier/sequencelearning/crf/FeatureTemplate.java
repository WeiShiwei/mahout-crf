package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.util.LineReader;

public class FeatureTemplate {

	ArrayList<String> unigram_templs=new ArrayList<String>();
	ArrayList<String> bigram_templs=new ArrayList<String>();
	
	public FeatureTemplate(String templateUri) throws IOException{
		Configuration conf = new Configuration();
		FileSystem fs = FileSystem.get(URI.create(templateUri), conf);
		FSDataInputStream dis = null;
		try {
			dis = fs.open(new Path(templateUri));
			LineReader in = new LineReader(dis,conf);  
			Text line = new Text();
			//按行读取
			while(in.readLine(line) > 0){
				String templateLine=line.toString();
				if(templateLine.startsWith("U")){
					unigram_templs.add(templateLine);
				}
				if(templateLine.startsWith("B")){
					bigram_templs.add(templateLine);
				}
			    System.out.println(templateLine);
			}
			dis.close();
			in.close();
		} finally {
			IOUtils.closeStream(dis);
		}
		
	}
}
