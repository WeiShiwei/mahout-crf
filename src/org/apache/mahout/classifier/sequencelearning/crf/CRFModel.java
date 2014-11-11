package org.apache.mahout.classifier.sequencelearning.crf;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

public class CRFModel {
	final double const_factor=1.0;
	
	/*语料库中提取的特征的最大数量，alpha和expected的维度都是maxid*/
	int maxid;
	/*特征的权重系数*/
	Vector alpha;
	/*特征的期望（模型期望与经验期望）*/
	Vector expected;
	/*目标函数值*/
	double obj;
	/*err统计的是当前迭代下，该线程总共预测token错误的个数*/
	int err;
	/*zeroone统计的是当前迭代下，该线程总共预测sentence错误的个数*/
	int zeroone;
	
	/*特征索引序列文件的URI*/
	String featureIndexSerializerURI;
	  
	/**
	 * 构造函数
	 * @param maxid
	 */
	public CRFModel(int maxid,String featureIndexSerializerURI){
		this.maxid = maxid;
		alpha = new DenseVector(maxid);
		expected = new DenseVector(maxid);
		obj=0;
		err=0;
		zeroone=0;
		this.featureIndexSerializerURI=featureIndexSerializerURI;
	}
	/**
	 * 构造函数（序列化）
	 * @param maxid
	 * @param alpha
	 * @param expected
	 * @param obj
	 * @param err
	 * @param zeroone
	 * @param featureIndexSerializerURI
	 */
	public CRFModel(int maxid,Vector alpha,Vector expected,double obj,int err,int zeroone,String featureIndexSerializerURI){
		this.maxid = maxid;
		this.alpha = alpha;
		this.expected = expected;
		this.obj = obj;
		this.err = err;
		this.zeroone = zeroone;
		this.featureIndexSerializerURI=featureIndexSerializerURI;
	}
//	public void setFeatureIndexerSerializerURI(String uri){
//		featureIndexSerializerURI=uri;
//	}
	
}