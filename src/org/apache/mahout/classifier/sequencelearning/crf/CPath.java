package org.apache.mahout.classifier.sequencelearning.crf;

import java.util.ArrayList;

import org.apache.mahout.math.Vector;

public class CPath {

	Node      rnode;
	Node      lnode;
	ArrayList<Integer>   fvector=new ArrayList<Integer>();
	double     cost;
	
	public CPath(){
		rnode=lnode=null;
		fvector=null;
		cost=0;
	}
	
	// for CRF
	void calcExpectation(Vector expected, double Z, int ysize){
		double c = Math.exp(lnode.alpha + cost + rnode.beta - Z);
		for(int f : fvector){
			expected.set((f+lnode.y*ysize+rnode.y), expected.get(f+lnode.y*ysize+rnode.y)+c);
		}
	}
	void add(Node _lnode, Node _rnode){
		lnode=_lnode;
		rnode=_rnode;
		lnode.rpath.add(this);
		rnode.lpath.add(this);
	}
	void clear() {
	    rnode = lnode = null;
	    fvector = null;
	    cost = 0;
	}
	///typedef std::vector<Path*>::const_iterator const_Path_iterator;
}
