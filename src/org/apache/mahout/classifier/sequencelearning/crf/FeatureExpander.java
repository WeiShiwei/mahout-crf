package org.apache.mahout.classifier.sequencelearning.crf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class FeatureExpander {
	private static int xsize;//标注语料库横向的维度,初始化后不能改变
	private static int index;//templateLine的字符索引
	static String BOS[] = { "_B-1", "_B-2", "_B-3", "_B-4"};
	static String EOS[] = { "_B+1", "_B+2", "_B+3", "_B+4"};
	
//	private int ysize;//此ysize指的是句子token的数量，他在FeatureStrGenerate()中设置
	private Set<String> HiddenStateSet=new HashSet<String>();//隐藏状态集合
	private ArrayList<String>  HiddenStateList=new ArrayList<String>();//隐藏状态列表
	
	FeatureTemplate featureTemplate;
	/**
	 * 构造函数
	 * @param featureTemplate
	 * featureTemplate是特征模板类的实例
	 */
	public FeatureExpander(FeatureTemplate featureTemplate,int xsize){
		this.featureTemplate=featureTemplate;//特征模板
		this.xsize=xsize;
	}
	
//	 /**
//	  * @return
//	  * 返回句子token的数量
//	  * 这个ysize有用吗
//	  */
//	public int getysize(){
//		return ysize;
//	}
	/**
	 * @return
	 * 返回句子的隐藏状态序列
	 */
	public ArrayList<String> getHiddenStateList(){
		return HiddenStateList;
	}
	
	/**
	 * @return
	 * 返回预测标识集合,用在FeatureIndexMapper
	 */
	public Set<String> getHiddenStateSet(){
		return HiddenStateSet;
	}
	
	/**
	 * 
	 * @param sentence
	 * @param x
	 * 特征二维列表，(tokenNUM*UnNUM)维数unigram矩阵和（tokenNUM*BiNUM）维数bigram矩阵竖向叠放在一起
	 * @param answer
	 * sentence的隐藏状态列表
	 * @return
	 */
	public boolean Expand(String sentence,TaggerImpl tagger){
		ArrayList<ArrayList<String> > tokenALAL=new ArrayList<ArrayList<String> >();
		
		int max_xsize=0;int min_xsize=999;
		
		String tokenArr[]=sentence.split("@@");	
		for(int i=0;i<tokenArr.length;i++){//tokenArr[i]="a1 b1 c1"
			String token[]=tokenArr[i].split(" ");//token={a1 b1 c1}
			tagger.answerStr.add(token[xsize-1]);/*answer*/
			
			if(token.length>max_xsize){max_xsize=token.length;}
			if(token.length<min_xsize){min_xsize=token.length;}
			
			ArrayList<String> tokenAL= new ArrayList<String>();
			for(int j=0;j<token.length;j++){
				tokenAL.add(token[j]);
			}
			tokenALAL.add(tokenAL);
		}
		/*验证判断*/
		if(max_xsize!=min_xsize||xsize>max_xsize){//如果条件成立，舍弃该句子（当然这种判断并不严谨，但暂时这样）
			System.out.println("ERROR：max_xsize!=min_xsize||xsize>max_xsize");
			return false;
		}
		
		//Unigram template
		for(int i=0;i<tokenALAL.size();i++){
			ArrayList<String> featureAL=new ArrayList<String>();
			for(int j=0;j<featureTemplate.unigram_templs.size();j++){
				StringBuffer feature=new StringBuffer();//
				if(!applyRule(feature,featureTemplate.unigram_templs.get(j),i,tokenALAL)){
					System.out.println("unigram applyRule error");
					return false;
				}
				featureAL.add(feature.toString());
			}
			tagger.xStr.add(featureAL);/*x*/
		}
		//Bigram template;
		for(int i=0;i<tokenALAL.size();i++){
			ArrayList<String> featureAL=new ArrayList<String>();
			for(int j=0;j<featureTemplate.bigram_templs.size();j++){
				StringBuffer feature=new StringBuffer();
				if(!applyRule(feature,featureTemplate.bigram_templs.get(j),i,tokenALAL)){
					System.out.println("bigram applyRule error");
					return false;
				}
				featureAL.add(feature.toString());
			}
			tagger.xStr.add(featureAL);/*x*/
		}
		
		return true;
	}

	/**
	 * 删除
	 * 扩展特征
	 * @param sentence
	 * {a1 b1 c1@@a2 b2 c2@@...}的形式
	 * @return
	 */	
	public ArrayList<String> Expand(String sentence){
		ArrayList<ArrayList<String> > tokenALAL=new ArrayList<ArrayList<String> >();
		ArrayList<String> featureAL=new ArrayList<String>();
		
		String tokenArr[]=sentence.split("@@");	
		
		int max_xsize=0;
		int min_xsize=999;
		for(int i=0;i<tokenArr.length;i++){//tokenArr[i]="a1 b1 c1"
			String token[]=tokenArr[i].split(" ");//token={a1 b1 c1}
			ArrayList<String> tokenAL= new ArrayList<String>();
			
			if(token.length>max_xsize){max_xsize=token.length;}
			if(token.length<min_xsize){min_xsize=token.length;}
			for(int j=0;j<token.length;j++){
				tokenAL.add(token[j]);
			}
			tokenALAL.add(tokenAL);
		}
		
		
		/*验证判断*/
		if(max_xsize!=min_xsize||xsize>max_xsize){//如果条件成立，舍弃该句子（当然这种判断并不严谨，但暂时这样）
			System.out.println("ERROR：max_xsize!=min_xsize||xsize>max_xsize");
			return null;
		}
		/*设置与此sentence相关联的数据*/
//		ysize=tokenALAL.size();
	    for(int i=0;i<tokenALAL.size();i++){
	    	HiddenStateList.add(tokenALAL.get(i).get(xsize-1));
	    	HiddenStateSet.add(tokenALAL.get(i).get(xsize-1));
	    }
	    
	    
	    
		//Unigram template
		for(int i=0;i<tokenALAL.size();i++){
			for(int j=0;j<featureTemplate.unigram_templs.size();j++){
				StringBuffer feature=new StringBuffer();//
				if(!applyRule(feature,featureTemplate.unigram_templs.get(j),i,tokenALAL)){
					System.out.println("unigram applyRule error");
				}
				//System.out.println("feature:"+feature);
				featureAL.add(feature.toString());
			}
		}
		
		//Bigram template;
		for(int i=0;i<tokenALAL.size();i++){
			for(int j=0;j<featureTemplate.bigram_templs.size();j++){
				StringBuffer feature=new StringBuffer();
				if(!applyRule(feature,featureTemplate.bigram_templs.get(j),i,tokenALAL)){
					System.out.println("bigram applyRule error");
				}
				//System.out.println("feature:"+feature);
				featureAL.add(feature.toString());
			}
		}
		
		return featureAL;
	}
	
	private boolean applyRule(StringBuffer feature,String tempLine,int pos,ArrayList<ArrayList<String> > sentenceALAL){
		//tempLine="U00:%x[-2,0]"	
		index=0;//index是templine的字符索引
		for(;index<tempLine.length();index++){
			switch (tempLine.charAt(index)){
				default:
					feature.append(tempLine.charAt(index));
					break;
				case '%':
					index++;
					switch (tempLine.charAt(index)){
						case 'x':
							index++;
							String r= getIndex(tempLine, pos, sentenceALAL);//pos, sentenceALAL
							//
							if(r==null){
								return false;
							}
							//System.out.println("r:"+r);
							feature.append(r);
							break;
						default:
							return false;
					}
				break;
				
			}
			
		}
		return true;
	}
	private static String getIndex(String tempLine,int pos,ArrayList<ArrayList<String> > sentenceALAL){
		if(tempLine.charAt(index)!='['){
			return null;
		}
		index++;
		
		int col = 0;
		int row = 0;
		int neg = 1;//neg
		if(tempLine.charAt(index)=='-'){
			neg = -1;
			index++;
		}
		
		for(;index<tempLine.length();index++){
			switch (tempLine.charAt(index)) {
				case '0': case '1': case '2': case '3': case '4':
		        case '5': case '6': case '7': case '8': case '9':
		        	row = 10 * row +(tempLine.charAt(index) - '0');
		        	break;
		        case ',':
		        	index++;
		        	//goto NEXT1;
		        	return NEXT1(tempLine, pos, sentenceALAL,row,col,neg);
		        default: return  null;
			}
		}
		return  null;	
	}
	private static  String NEXT1(String tempLine,int pos,ArrayList<ArrayList<String> > sentenceALAL,int row,int col,int neg){
		//NEXT1
		for(;index<tempLine.length();index++){
			switch (tempLine.charAt(index)) {
				case '0': case '1': case '2': case '3': case '4':
		        case '5': case '6': case '7': case '8': case '9':
		        	col = 10 * col +(tempLine.charAt(index) - '0');
		        	break;
		        case ']':
		        {
		        	//NEXT2
		        	row *= neg;
		        	//例：浅层分析中col={0,1,2},xsize=3,xsize=col+1;所以col >=3=(MaxOFcol+1)=(xsize-1+1),条件可以改为col>xsize
		    		if (row < -4 || row > 4 ||col < 0 || col >=xsize ) {////\static_cast<int>(tagger.xsize())=3
		    			return null;
		    		}
		    		
		    		// TODO(taku): very dirty workaround
		    		//if (check_max_xsize_) {
		    			//max_xsize_ = std::max(max_xsize_, static_cast<unsigned int>(col + 1));
		    		//}

		    		int idx = pos + row;
		    		if (idx < 0) {
		    			return BOS[-idx-1];
		    		}
		    		if (idx >= sentenceALAL.size()) {////\sentenceALAL.size()=tagger.size()
		    		    return EOS[idx - sentenceALAL.size()];////\sentenceALAL.size()=tagger.size()
		    		}
		    		
		    		return sentenceALAL.get(idx).get(col);
		        }
		        	
		        default: return  null;
			}
		}
		return null;//
	}
}
