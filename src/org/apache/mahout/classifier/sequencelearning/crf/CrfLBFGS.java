package org.apache.mahout.classifier.sequencelearning.crf;

import riso.numerical.LBFGS;

public class CrfLBFGS {
	
	int m=5;
	int[] iprint ;
	boolean diagco;
	double diag [ ]; 
	double eps, xtol;
	
	public void optimize ( int n ,  double[] x , double f , double[] g , int[] iflag){
		if(diag==null){
			m=5;
			iprint = new int [ 2 ];
			iprint [ 1 -1] = 1;
			iprint [ 2 -1] = 0;
			
			diagco=false;
			diag= new double [ n ];
			eps= 1.0e-5;
			xtol= 1.0e-16;
		}
		/**************************************A调试**********************************************/
		
		System.out.println("------TaggerImplReducer------");
		System.out.println("x详细信息：");
		for(int hh=0;hh<x.length;hh++){
			System.out.println("x["+hh+"]	"+x[hh]);
		}
		System.out.println();
		System.out.println("f="+f);
		
		System.out.println("g详细信息：");
		for(int hh=0;hh<g.length;hh++){
			System.out.println("g["+hh+"]	"+g[hh]);
		}		

//		System.out.println("diag详细信息：");
//		for(int hh=0;hh<diag.length;hh++){
//			System.out.println("diag["+hh+"]	"+diag[hh]);
//		}
		/**************************************B**********************************************/
		/*lbfgs*/
		try{
			LBFGS.lbfgs ( n , m , x , f , g , diagco , diag , iprint , eps , xtol , iflag );
		}
		catch (LBFGS.ExceptionWithIflag e){
			System.err.println( "Sdrive: lbfgs failed.\n"+e );
			return;
		}
	}

	
	/**
	 * 构造函数（用于序列化）
	 * @param m
	 * @param iprint
	 * @param diagco
	 * @param diag
	 * @param eps
	 * @param xtol
	 */
	public CrfLBFGS(int m,int[] iprint,boolean diagco,double[] diag,double eps,double xtol){
		this.m=m;
		this.iprint=iprint;
		this.diagco=diagco;
		this.diag=diag;
		this.eps=eps;
		this.xtol=xtol;
	}
	public CrfLBFGS(){}
}



///*lbfgs*/
//int ndim=model.maxid;int msave=7;
//int nwork=ndim * ( 2 * msave + 1 ) + 2 * msave ;
//double x [ ] , g [ ] , diag [ ] , w [ ];
//x = new double [ ndim ];///
//g = new double [ ndim ];///
//diag = new double [ ndim ];
//w = new double [ nwork ];
//
//double f, eps, xtol, gtol, t1, t2, stpmin, stpmax;
//int iprint [ ] , iflag[] = new int[1], icall, n, m, mp, lp, j;
//iprint = new int [ 2 ];
//boolean diagco;
//
////n=model.maxid;//{x0,x1...xn}共有n+1个x
////m=5;
//
//iprint [ 1 -1] = 1;
//iprint [ 2 -1] = 0;
//diagco= false;
//eps= 1.0e-5;
//xtol= 1.0e-16;
//icall=0;
//iflag[0]=0;
