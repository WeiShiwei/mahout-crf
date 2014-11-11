package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

public class LossyCRFModelSerializer {

	public LossyCRFModelSerializer(){
	}
	
	static void serialize(CRFModel model, DataOutput output) throws IOException {
		int maxid=model.maxid;
		Vector alpha=model.alpha;VectorWritable alphaW=new VectorWritable(alpha);
		Vector expected=model.expected;VectorWritable expectedW=new VectorWritable(expected);
		double obj=model.obj;
		int err=model.err;
		int zeroone=model.zeroone;
		String featureIndexSerializerURI=model.featureIndexSerializerURI;
		
		output.writeInt(maxid);
		alphaW.write(output);
		expectedW.write(output);
		output.writeDouble(obj);
		output.writeInt(err);
		output.writeInt(zeroone);
		output.writeUTF(featureIndexSerializerURI);
	}
	
	static CRFModel deserialize(DataInput input) throws IOException {
		int maxid=input.readInt();
		Vector alpha=VectorWritable.readVector(input);
		Vector expected=VectorWritable.readVector(input);
		double obj=input.readDouble();
		int err=input.readInt();
		int zeroone=input.readInt();
		String featureIndexSerializerURI=input.readUTF();
		
		return new CRFModel(maxid,alpha,expected,obj,err,zeroone,featureIndexSerializerURI);
	}
}
