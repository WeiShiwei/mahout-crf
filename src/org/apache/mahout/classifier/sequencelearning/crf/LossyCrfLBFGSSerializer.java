package org.apache.mahout.classifier.sequencelearning.crf;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import riso.numerical.LBFGS;
import riso.numerical.Mcsrch;

public class LossyCrfLBFGSSerializer {

	static void serialize(CrfLBFGS lbfgs, DataOutput output) throws IOException {
		
		int m=lbfgs.m;//m
		
		int[] iprint=lbfgs.iprint;//iprint
		Vector iprintVec=new DenseVector(iprint.length);
		for(int i=0;i<iprintVec.size();i++){
			iprintVec.set(i, iprint[i]);
		}
		VectorWritable iprintVecWritable=new VectorWritable(iprintVec);
		
		boolean diagco=lbfgs.diagco;//diagco
		
		double diag[] =lbfgs.diag;//diag
		Vector diagVec=new DenseVector(diag.length);
		diagVec.assign(diag);
		VectorWritable diagVecWritable=new VectorWritable(diagVec);
		
		double eps =lbfgs.eps;//eps
		double xtol=lbfgs.xtol;//xtol
		
		output.writeInt(m);
		iprintVecWritable.write(output);
		output.writeBoolean(diagco);
		diagVecWritable.write(output);
		output.writeDouble(eps);
		output.writeDouble(xtol);
		
		////////////////////////////////////////////LBFGS的序列化//////////////////////////////////////////////////////////////
		double gtol=LBFGS.gtol;
		double stpmin=LBFGS.stpmin;
		double stpmax=LBFGS.stpmax;
		
		boolean solution_cacheIsNull;
		VectorWritable solution_cacheVecWritable=new VectorWritable();
		if(LBFGS.solution_cache!=null){
			solution_cacheIsNull=false;
			double[] solution_cache=LBFGS.solution_cache;
			Vector solution_cacheVec=new DenseVector(solution_cache.length);
			for(int i=0;i<solution_cacheVec.size();i++){
				solution_cacheVec.set(i, solution_cache[i]);
			}
			solution_cacheVecWritable=new VectorWritable(solution_cacheVec);
		}else{
			solution_cacheIsNull=true;
		}
		
		/*****************LBFGS(1)****************/
		output.writeDouble(gtol);
		output.writeDouble(stpmin);
		output.writeDouble(stpmax);
		output.writeBoolean(solution_cacheIsNull);
		if(solution_cacheIsNull==false){
			solution_cacheVecWritable.write(output);
		}		
		/*********************************/
		double gnorm = LBFGS.gnorm;
		double stp1 = LBFGS.stp1;
		double ftol = LBFGS.ftol;
		double stp_0Array= LBFGS.stp[0];		
		double ys = LBFGS.ys;
		double yy = LBFGS.yy;
		double sq = LBFGS.sq;
		double yr = LBFGS.yr;
		double beta = LBFGS.beta;
		double xnorm = LBFGS.xnorm;
		/***************LBFGS(2)******************/
		output.writeDouble(gnorm);
		output.writeDouble(stp1);
		output.writeDouble(ftol);
		output.writeDouble(stp_0Array);
		output.writeDouble(ys);
		output.writeDouble(yy);
		output.writeDouble(sq);
		output.writeDouble(yr);
		output.writeDouble(beta);
		output.writeDouble(xnorm);
		/*********************************/
		int iter = LBFGS.iter;
		int nfun = LBFGS.nfun;
		int point = LBFGS.point;
		int ispt = LBFGS.ispt;
		int iypt = LBFGS.iypt;
		int maxfev = LBFGS.maxfev;		
		int info_0Array = LBFGS.info[0];
		int bound = LBFGS.bound;
		int npt = LBFGS.npt;
		int cp = LBFGS.cp;
		int i = LBFGS.i;
		int nfev_0Array = LBFGS.nfev[0];
		int inmc = LBFGS.inmc;
		int iycn = LBFGS.iycn;
		int iscn = LBFGS.iscn;		
		boolean finish = LBFGS.finish;
		/**************LBFGS(3)*******************/
		output.writeInt(iter);
		output.writeInt(nfun);
		output.writeInt(point);
		output.writeInt(ispt);
		output.writeInt(iypt);
		output.writeInt(maxfev);
		output.writeInt(info_0Array);
		output.writeInt(bound);
		output.writeInt(npt);
		output.writeInt(cp);
		output.writeInt(i);
		output.writeInt(nfev_0Array);
		output.writeInt(inmc);
		output.writeInt(iycn);
		output.writeInt(iscn);
		output.writeBoolean(finish);
		/*********************************/
		
        ////////////////////////////////////////////Mcsrch的序列化//////////////////////////////////////////////////////////////
		int infoc_0Array=Mcsrch.infoc[0];		
		int j = Mcsrch.j;
		/****************Mcsrch1*****************/
		output.writeInt(infoc_0Array);
		output.writeInt(j);
		/*********************************/
		double dg = Mcsrch.dg;
		double dgm = Mcsrch.dgm;
		double dginit = Mcsrch.dginit;
		double dgtest = Mcsrch.dgtest;
//		double dgx[] = new double[1];
		//		double dgxm[] = new double[1];
		//		double dgy[] = new double[1];
		//		double dgym[] = new double[1];
		double dgx_0Array = Mcsrch.dgx[0];
		double dgxm_0Array = Mcsrch.dgxm[0];
		double dgy_0Array = Mcsrch.dgy[0];
		double dgym_0Array = Mcsrch.dgym[0];
		double finit = Mcsrch.finit;
		double ftest1 = Mcsrch.ftest1;
		double fm = Mcsrch.fm;
        //		double fx[] = new double[1];
		//		double fxm[] = new double[1];
		//		double fy[] = new double[1];
		//		double fym[] = new double[1];
		double fx_0Array = Mcsrch.fx[0];
		double fxm_0Array = Mcsrch.fxm[0];
		double fy_0Array = Mcsrch.fy[0];
		double fym_0Array = Mcsrch.fym[0];
		double p5 = Mcsrch.p5;
		double p66 = Mcsrch.p66;
//		double stx[] = new double[1];
//		double sty[] = new double[1];
		double stx_0Array = Mcsrch.stx[0];
		double sty_0Array = Mcsrch.sty[0];
		double stmin = Mcsrch.stmin;
		double stmax = Mcsrch.stmax;
		double width = Mcsrch.width;
		double width1 = Mcsrch.width1;
		double xtrapf = Mcsrch.xtrapf;
		/****************Mcsrch2*****************/
		output.writeDouble(dg);
		output.writeDouble(dgm);
		output.writeDouble(dginit);
		output.writeDouble(dgtest);
		output.writeDouble(dgx_0Array);
		output.writeDouble(dgxm_0Array);
		output.writeDouble(dgy_0Array);
		output.writeDouble(dgym_0Array);
		output.writeDouble(finit);
		output.writeDouble(ftest1);
		output.writeDouble(fm);
		output.writeDouble(fx_0Array);
		output.writeDouble(fxm_0Array);
		output.writeDouble(fy_0Array);
		output.writeDouble(fym_0Array);
		output.writeDouble(p5);
		output.writeDouble(p66);
		output.writeDouble(stx_0Array);
		output.writeDouble(sty_0Array);
		output.writeDouble(stmin);
		output.writeDouble(stmax);
		output.writeDouble(width);
		output.writeDouble(width1);
		output.writeDouble(xtrapf);
		/*********************************/
		
		boolean brackt_0Array = Mcsrch.brackt[0];
		boolean stage1 = Mcsrch.stage1;
		/***************Mcsrch3******************/
		output.writeBoolean(brackt_0Array);
		output.writeBoolean(stage1);
		/*********************************/
	}
	
	static CrfLBFGS deserialize(DataInput input) throws IOException {
		int m=input.readInt();
		
		Vector iprintVec=VectorWritable.readVector(input);
		int[] iprint=new int[iprintVec.size()];
		for(int i=0;i<iprint.length;i++){
			iprint[i]=(int)iprintVec.get(i);
		}
		
		boolean diagco=input.readBoolean();
		
		Vector diagVec=VectorWritable.readVector(input);
		double[] diag=new double[diagVec.size()];
		for(int i=0;i<diag.length;i++){
			diag[i]=diagVec.get(i);
		}
		
		double eps=input.readDouble();
		double xtol=input.readDouble();
		
		////////////////////////////////////////////LBFGS的序列化//////////////////////////////////////////////////////////////
		/*****************LBFGS(1)****************/
		LBFGS.gtol=input.readInt();
		LBFGS.stpmin=input.readInt();
		LBFGS.stpmax=input.readInt();
		boolean solution_cacheIsNull=input.readBoolean();
		if(solution_cacheIsNull==false){
			Vector solution_cacheVec=VectorWritable.readVector(input);
			double[] solution_cache=new double[solution_cacheVec.size()];
			for(int i=0;i<solution_cache.length;i++){
				solution_cache[i]=(double)solution_cacheVec.get(i);
			}
			LBFGS.solution_cache=solution_cache;
		}else{
			LBFGS.solution_cache=null;
		}
		
		/*********************************/
		/***************LBFGS(2)******************/
		LBFGS.gnorm=input.readDouble();
		LBFGS.stp1=input.readDouble();
		LBFGS.ftol=input.readDouble();
		LBFGS.stp[0]=input.readDouble();
		LBFGS.ys=input.readDouble();
		LBFGS.yy=input.readDouble();
		LBFGS.sq=input.readDouble();
		LBFGS.yr=input.readDouble();
		LBFGS.beta=input.readDouble();
		LBFGS.xnorm=input.readDouble();
		/*********************************/
		/***************LBFGS(3)******************/
		LBFGS.iter=input.readInt();
		LBFGS.nfun=input.readInt();
		LBFGS.point=input.readInt();
		LBFGS.ispt=input.readInt();
		LBFGS.iypt=input.readInt();
		LBFGS.maxfev=input.readInt();
		LBFGS.info[0]=input.readInt();		
		LBFGS.bound=input.readInt();
		LBFGS.npt=input.readInt();
		LBFGS.cp=input.readInt();
		LBFGS.i=input.readInt();
		LBFGS.nfev[0]=input.readInt();
		LBFGS.inmc=input.readInt();
		LBFGS.iycn=input.readInt();
		LBFGS.iscn=input.readInt();
		LBFGS.finish=input.readBoolean();
		/*********************************/
		
        ////////////////////////////////////////////Mcsrch的序列化//////////////////////////////////////////////////////////////
		/****************Mcsrch1*****************/
		Mcsrch.infoc[0]=input.readInt();
		Mcsrch.j=input.readInt();
		/*********************************/
		/****************Mcsrch2*****************/
		Mcsrch.dg=input.readDouble();
		Mcsrch.dgm=input.readDouble();
		Mcsrch.dginit=input.readDouble();
		Mcsrch.dgtest=input.readDouble();
		Mcsrch.dgx[0]=input.readDouble();
		Mcsrch.dgxm[0]=input.readDouble();
		Mcsrch.dgy[0]=input.readDouble();
		Mcsrch.dgym[0]=input.readDouble();
		Mcsrch.finit=input.readDouble();
		Mcsrch.ftest1=input.readDouble();
		Mcsrch.fm=input.readDouble();
		
		Mcsrch.fx[0]=input.readDouble();
		Mcsrch.fxm[0]=input.readDouble();
		Mcsrch.fy[0]=input.readDouble();
		Mcsrch.fym[0]=input.readDouble();
		Mcsrch.p5=input.readDouble();
		Mcsrch.p66=input.readDouble();
		Mcsrch.stx[0]=input.readDouble();
		Mcsrch.sty[0]=input.readDouble();
		Mcsrch.stmin=input.readDouble();
		Mcsrch.stmax=input.readDouble();
		Mcsrch.width=input.readDouble();
		Mcsrch.width1=input.readDouble();
		Mcsrch.xtrapf=input.readDouble();
		/*********************************/
		/***************Mcsrch3******************/
		Mcsrch.brackt[0]=input.readBoolean();
		Mcsrch.stage1=input.readBoolean();
		/*********************************/
		
	
		return new CrfLBFGS(m,iprint,diagco,diag,eps,xtol);
	}
	
}
