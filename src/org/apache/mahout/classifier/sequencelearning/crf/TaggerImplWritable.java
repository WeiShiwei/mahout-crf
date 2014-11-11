package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import org.apache.hadoop.io.Writable;

public class TaggerImplWritable implements Writable{
	private TaggerImpl taggerImpl;
	static String Delimiter="@@@";
	
	public TaggerImpl getValue() {
	    return taggerImpl;
	}
	  
	public void setValue(TaggerImpl value) {
	    this.taggerImpl = value;
	}
	
	@Override
	  public void write(DataOutput out) throws IOException {//PolymorphicWritable多态可写
	    //PolymorphicWritable.write(out, value);
		writeTaggerImpl(out, taggerImpl);
	  }
	  
	  @Override
	  public void readFields(DataInput in) throws IOException {
	    //value = PolymorphicWritable.read(in, Cluster.class);
		  taggerImpl = readTaggerImpl(in);
	  }
	  
	  public static String ListToString(ArrayList<String> stringList){
		  if (stringList==null) {
			  return null;
		  }
		  StringBuilder result=new StringBuilder();
		  boolean flag=false;
		  for (String string : stringList) {
			  if (flag) {
				  result.append(Delimiter);
			  }else {
				  flag=true;
			  }
			  result.append(string);
		  }
		  return result.toString();
	  }
	  
	  
	  /** Writes a typed TaggerImpl instance to the output stream */
	  public static void writeTaggerImpl(DataOutput out, TaggerImpl taggerImpl) throws IOException {
		  //正确用法String[] arrString = (String[])list.toArray(new String[0]);
		  
////		  int vticalNum=taggerImpl.xStr.size();
//////		  out.writeInt(vticalNum);
////		  
////		  int tokenNum=taggerImpl.answerStr.size();
//////		  out.writeInt(tokenNum);
////		  
////		  for(int i=0;i<vticalNum;i++){
////			  String flist=ListToString(taggerImpl.xStr.get(i));
////			  out.writeUTF(flist);
////		  }
//		  \
		  
		  //写vticalNum次
		  int vticalNum=taggerImpl.xStr.size();
		  out.writeInt(vticalNum);
		  for(ArrayList<String> flist : taggerImpl.xStr){
			  String list=ListToString(flist);
			  out.writeUTF(list);
		  }
		  //写1次
		  String list=ListToString(taggerImpl.answerStr);
		  out.writeUTF(list);
		  
//		  System.out.println("taggerImpl.x.size():"+taggerImpl.x.size());///=0???
		  if(!taggerImpl.x.isEmpty()){
			  out.writeBoolean(true);
			  //写vticalNum次
			  for(ArrayList<Integer> flist_I : taggerImpl.x){
//				  System.out.println("writeTaggerImpl(flist_I):"+flist_I);///
				  ArrayList<String> flist_S=new ArrayList<String>();
				  for(Integer i : flist_I){
					  flist_S.add(String.valueOf(i));
				  }
				  
				  String flist=ListToString(flist_S);
//				  System.out.println("writeTaggerImpl(flist):"+flist);///
				  out.writeUTF(flist);
			  }
		  }else{
			  out.writeBoolean(false);
		  }
		  
		  
		  if(!taggerImpl.answer.isEmpty()){
			  out.writeBoolean(true);
			  //写1次
			  ArrayList<String> answerlist=new ArrayList<String>();
			  for(Integer i : taggerImpl.answer){
				  answerlist.add(String.valueOf(i));
			  }
			  String answer=ListToString(answerlist);
			  out.writeUTF(answer);
		  }else{
			  out.writeBoolean(false);  
		  }
		  
		  out.writeInt(taggerImpl.xsize);
		  out.writeInt(taggerImpl.ysize);
	  }
	  
	  /** Read a typed TaggerImpl instance from the input stream */
	  public static TaggerImpl readTaggerImpl(DataInput in) throws IOException {
		  int vticalNum = in.readInt();//
		  //读vticalNum次
		  ArrayList<ArrayList<String> > xStr=new ArrayList<ArrayList<String> >();
		  for(int i=0;i<vticalNum;i++){
			  String[] farray=in.readUTF().split(Delimiter);
			  ArrayList<String> flist=new ArrayList<String>(Arrays.asList(farray));
			  xStr.add(flist);
		  }
		  //读1次
		  String[] sarray=in.readUTF().split(Delimiter);
		  ArrayList<String> answerStr=new ArrayList<String>(Arrays.asList(sarray));
		  
////		//读vticalNum次
////		  System.out.println("in.readUTF()="+in.readUTF());///
////		  ArrayList<ArrayList<Integer> > x=new ArrayList<ArrayList<Integer> >();
////		  for(int i=0;i<vticalNum;i++){
////			  String[] farray=in.readUTF().split(Delimiter);
////			  ArrayList<Integer> flist=new ArrayList<Integer>();
////			  System.out.println("readTaggerImpl:flist="+flist);///
////			  for(int j=0;j<farray.length;j++){
////				  flist.add(Integer.parseInt(farray[j]));
////			  }
////			  x.add(flist);
////		  }
//		//读1次
////		  String[] sarray_answer=in.readUTF().split(Delimiter);
////		  ArrayList<Integer> answer=new ArrayList<Integer>();
////		  for(int j=0;j<sarray_answer.length;j++){
////			  answer.add(Integer.parseInt(sarray_answer[j]));
////		  }
//		  \
		  
		  boolean x_bool=in.readBoolean();
		  ArrayList<ArrayList<Integer> > x=new ArrayList<ArrayList<Integer> >();
		  if(x_bool){
			  //读vticalNum次
			  for(int i=0;i<vticalNum;i++){
				  String[] farray=in.readUTF().split(Delimiter);
				  ArrayList<Integer> flist=new ArrayList<Integer>();
				  for(int j=0;j<farray.length;j++){
					  flist.add(Integer.parseInt(farray[j]));
				  }
				  x.add(flist);
			  }
		  }
		  
		  boolean answer_bool=in.readBoolean();
		  ArrayList<Integer> answer=new ArrayList<Integer>();
		  if(answer_bool){
			//读1次
			  String[] sarray_answer=in.readUTF().split(Delimiter);
			  for(int j=0;j<sarray_answer.length;j++){
				  answer.add(Integer.parseInt(sarray_answer[j]));
			  }
		  }
		  
		  int xsize=in.readInt();
		  int ysize=in.readInt();
		  
		  TaggerImpl taggerImpl=new TaggerImpl(xStr,answerStr,x,answer,xsize,ysize);
		  return taggerImpl;
	  }
	
	
	
	
}