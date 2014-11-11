package org.apache.mahout.classifier.sequencelearning.crf;

import java.util.ArrayList;

import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;

/*读取TaggerImplWritable,然后反序列化*/
public class TaggerImpl {
	double cost_factor=1.0;
	
	final double LOG2 = 0.69314718055;//ln2=0.69314718055
	final double MINUS_LOG_EPSILON=50;//EPSILON希腊语字母之第五字
	double logsumexp(double x, double y, boolean flg) {
		if (flg) return y;  // init mode
		double vmin = Math.min(x, y);
		double vmax = Math.max(x, y);
		if (vmax > vmin + MINUS_LOG_EPSILON) {///MINUS_LOG_EPSILON 像一个阈值
			return vmax;
		} else {
			return vmax + Math.log(Math.exp(vmin - vmax) + 1.0);///【平滑处理？？？】
		}
	}

	/****************************TaggerImplWritable中需要存储的**********************************************/
	//xStr是x的String版本
	ArrayList<ArrayList<String> > xStr=new ArrayList<ArrayList<String> >();
	//answerStr是answer的String版本
	ArrayList<String> answerStr=new ArrayList<String>();
	/*特征索引矩阵x的元素都是fvector，前xsize个是状态特征的，后xsize个是转移特征的；因此x是（xsize*2）*（col）维,按行遍历*/
	ArrayList<ArrayList<Integer> > x=new ArrayList<ArrayList<Integer> >();
	/*token对应的标识序列 */
	ArrayList<Integer> answer=new ArrayList<Integer>();
	/*句子token的数量*/
	int xsize=0;
	/*预测标记(隐藏状态)集合的大小,不是句子token扩展出来的特征的数量，切记*/
	int ysize=0;
	
	/****************************TaggerImplWritable中不需要存储的，是计算得来的**********************************************/
	/*罚函数*/
	ArrayList<ArrayList<Double> > penalty=new ArrayList<ArrayList<Double> >();
	/*预测结果*/
	ArrayList<Integer> result=new ArrayList<Integer>();
	/**/
	int nbest;
	/**/
	double cost;
	/**/
	double Z;
	
	/**
	 * lattic网格
	 * nodeList共有（xsize）*（ysize）个节点，可以画一个二维的坐标轴，坐标（x,y）与索引index的对应关系是index=x*ysize+y
	 * pathList共有（xsize-1）*(ysize*ysize)个路径，可以尝试在nodeList中链接所有的节点，循环中（cur:2->xsize，j,k）对应的索引index=(cur-1)*ysize*ysize+j*ysize+k
	 */
	ArrayList<Node> nodeList;
	ArrayList<CPath> pathList;
	
	/**alpha与expected两者，只是在这里定义，其引用的是mapper中为两者分配alpha与expected*/
	/*特征的权重系数*/
	Vector alpha=new DenseVector();
	/*特征的期望（模型期望与经验期望）*/
	Vector expected=new DenseVector();
	
	
	/**
	 * 构造函数(序列化)
	 * @param xStr
	 * @param answerStr
	 * @param x
	 * @param y
	 * @param xsize
	 * @param ysize
	 */
	public TaggerImpl(ArrayList<ArrayList<String> > xStr,ArrayList<String> answerStr,
			ArrayList<ArrayList<Integer> > x,ArrayList<Integer> answer,int xsize,int ysize){
		this.xStr=xStr;
		this.answerStr=answerStr;
		this.x=x;
		this.answer=answer;
		this.xsize=xsize;
		this.ysize=ysize;
	}
	public TaggerImpl(){}
	
	/**
	 * buildLattice
	 * 创建网格（节点与边）
	 */
	private void buildLattice() {
		LatticAllocate();//为nodeList和pathList初始化和分配空间
		
		if(x.isEmpty()){
			return ;
		}
		//构建网格（节点和边）lattic
		int fid=0;
		for(int cur=0;cur<xsize;cur++){//节点
			ArrayList<Integer> fvector=x.get(fid++);
			for(int j=0;j<ysize;j++){
//				System.out.println("("+cur+","+j+")");
				Lattic(cur,j).set(cur,j,fvector);
			}
		}
//		System.out.println("node设置完成");
		for(int cur=1;cur<xsize;cur++){//路径
			ArrayList<Integer> fvector=x.get(fid++);
			for(int j=0;j<ysize;j++){
				for(int k=0;k<ysize;k++){
//					System.out.println("k:"+k);
					Lattic(cur,j,k).add(Lattic(cur-1,j), Lattic(cur,k));//void Path::add(Node _lnode, Node _rnode)注意对象参数是引用类型的
					Lattic(cur,j,k).fvector=fvector;
				}
			}
		}
//		System.out.println("path设置完成");
		//计算节点和路径的cost
		for(int cur=0;cur<xsize;cur++){
			for(int j=0;j<ysize;j++){
				calcCost(Lattic(cur,j));//节点cost
				for(int pindex=0;pindex<Lattic(cur,j).lpath.size();pindex++){
					calcCost(Lattic(cur,j).lpath.get(pindex));//路径cost
				}
			}
		}

	}
	
	/**
	 * LatticAllocate
	 * 为nodeList，pathList和result分配初始工作空间
	 */
	private void LatticAllocate(){
		//注意：ArrayList初始化即分配空间，提高效率
		int nodeNum=xsize*ysize;
//		System.out.println("LatticAllocate()  nodeNum:"+nodeNum);
		nodeList=new ArrayList<Node>(nodeNum);
		for(int i=0;i<nodeNum;i++){
			nodeList.add(new Node());
		}
		
		int pathNum=(xsize-1)*ysize*ysize;
//		System.out.println("LatticAllocate()  pathNum:"+pathNum);
		pathList=new ArrayList<CPath>(pathNum);
		for(int i=0;i<pathNum;i++){
			pathList.add(new CPath());
		}
		
		for(int i=0;i<xsize;i++){//result放在这里好像不太合适
			result.add(0);
		}
	}
	
	/**
	 * 计算节点的cost
	 * @param n
	 */
	private void calcCost(Node n) {
		double c=0;
		ArrayList<Integer> fvector=n.fvector;
		for(int f : fvector){
			c +=alpha.get(f+n.y);
		}
		n.cost =cost_factor *c;
	}
	/**
	 * 计算路径的cost
	 * @param p
	 */
	private void calcCost(CPath p) {
		double c=0;
		ArrayList<Integer> fvector=p.fvector;
		for(int f : fvector){
			c +=alpha.get(f+p.lnode.y*ysize+p.rnode.y);
		}
		p.cost =cost_factor *c;
	}
	
	/**
	 * 前向后向算法
	 */
	private void forwardbackward() {
		if(x.isEmpty()){
			return;
		}
		
		for(int i=0;i<xsize;i++){//计算节点的alpha值
			for(int j=0;j<ysize;j++){
				Lattic(i,j).calcAlpha();//【】
			}
		}
		
		for(int i=xsize-1;i>=0;i--){//计算节点的beta值，需反向计算
			for(int j=0;j<ysize;j++){
				Lattic(i,j).calcBeta();//【】
			}
		}
		
		Z=0.0;
		for (int j = 0; j < ysize; ++j){//根据节点的beta值，计算归一化因子Z
			Z = logsumexp(Z, Lattic(0,j).beta, j == 0);//【】
		}
	}
	/**
	 * viterbi算法
	 */
	private void viterbi() {
		for(int i=0;i<xsize;i++){//token的遍历
			for(int j=0;j<ysize;j++){//纵向的遍历
				double bestc = -1e37;
				Node best = null;
				for(CPath path : Lattic(i,j).lpath){
					double cost = path.lnode.bestCost+path.cost+Lattic(i,j).cost;
					if (cost > bestc) {
				          bestc = cost;
				          best  = path.lnode;
					}
				}
				Lattic(i,j).prev      =best;//最优路径上当前节点的前驱
				Lattic(i,j).bestCost  =(best!=null) ? bestc :Lattic(i,j).cost;//最优路径上当前节点的bestCost
			}
			
		}
		
		double bestc = -1e37;
		Node best = null;
		int s = xsize-1;
		for(int j=0;j<ysize;j++){
			if(bestc < Lattic(s,j).bestCost){
				best=Lattic(s,j);
				bestc=Lattic(s,j).bestCost;
			}
		}
		for(Node n=best;n!=null;n=n.prev){
			result.set(n.x, n.y);//
		}
		cost=-Lattic(xsize-1,result.get(xsize-1)).bestCost;//
		
		System.out.println();
		System.out.print("viterbi()预测的状态有序集合:");
		for(int thenode : result){
			System.out.print(thenode+" ");
		}
		System.out.println();
		
	}
	/**
	 * eval()
	 * @return
	 */
	public int eval() {
		// TODO Auto-generated method stub
		int err = 0;
		for (int i = 0; i < xsize; ++i) {
			if(answer.get(i)!=result.get(i)){
				++err;
			}
		}
		return err;
	}
	
	/**
	 * 计算梯度
	 * @return
	 * 返回obj
	 */
	public double gradient() {
		if(x.isEmpty()){
			return 0.0;
		}
		 
		buildLattice();
		System.out.println();
		System.out.println("buildLattice():");
//		NodeDebug();///调试
//		CPathDebug();
//		ExpectationDebug();///调试
		
		forwardbackward();
		System.out.println();
		System.out.println("forwardbackward():");
//		NodeDebug();///调试
//		CPathDebug();
//		ExpectationDebug();///调试
		
		/*********************************************************************/
		//因该说整个过程node节点的（x,y,alpha,beta,cost,nbestcost）与crf++的输出一致
		//在下面的calcExpectation开始，expected开始不一致了
		/*********************************************************************/
		
		double s = 0.0;
		//
		for(int i=0;i<xsize;i++){
			for(int j=0;j<ysize;j++){
				Lattic(i,j).calcExpectation(expected, Z, ysize);//【节点node和边path的期望】
			}
		}
		System.out.println();
		System.out.println("模型期望Lattic(i,j).calcExpectation(expected, Z, ysize):");
//		NodeDebug();///调试
//		CPathDebug();
//		ExpectationDebug();///调试
		
		for(int i=0;i<xsize;i++){
//			System.out.println("(i,answer.get(i))=: ("+i+","+answer.get(i)+")");///调试
			Node selectedNode=Lattic(i,answer.get(i));//selectedNode
			
			ArrayList<Integer> fvector=selectedNode.fvector;
//			System.out.println("fvector="+fvector);///调试
			
			for(int f : fvector){
				int index=f+answer.get(i);//expected的索引
//				System.out.println("index="+index);///调试
				expected.set(index, expected.get(index)-1);
			}
			s+=selectedNode.cost;
			
			ArrayList<CPath> pathAL=selectedNode.lpath;
			for(int j=0;j<pathAL.size();j++){
				Node lnode=pathAL.get(j).lnode;//该路径pathAL.get(j)的左边节点
				Node rnode=pathAL.get(j).rnode;//该路径pathAL.get(j)的右边节点
				if(lnode.y==answer.get(lnode.x)){//pathAL.get(j).lnode的x和y(条件是y=answer[x])
					ArrayList<Integer> pvector=pathAL.get(j).fvector;//这里是路径的fvector
					for(int f : pvector){
						int index=f+lnode.y*ysize+rnode.y;//expected的索引
						expected.set(index, expected.get(index)-1);
					}
					s+=pathAL.get(j).cost;
					break;
				}
			}
			
		}
		System.out.println();
		System.out.println("经验期望：");
//		NodeDebug();///调试
//		CPathDebug();
//		ExpectationDebug();///调试

		System.out.println();
		System.out.println("viterbi算法：");
		viterbi();
		
		System.out.println("Z="+Z);
		System.out.println("s="+s);
		System.out.println("(Z - s)="+(Z - s));
		return Z - s ;
	}
	
	/**
	 * 
	 * @param x
	 * 横轴坐标：0->xsize
	 * @param y
	 * 纵轴坐标：0->yszie
	 * @return
	 * 返回Node的引用
	 */
	private Node Lattic(int x,int y){
		//Node是对象，传递的是引用
		return nodeList.get(x*ysize+y);
	}
	/**
	 * 
	 * @param cur
	 * cur:1->xsize
	 * @param j
	 * j:0->ysize
	 * @param k
	 * y:0->ysize
	 * @return
	 */
	private CPath Lattic(int cur,int j,int k){
		//Path是对象，传递的是引用
		return pathList.get((cur-1)*ysize*ysize+j*ysize+k);
	}
	/**
	 * 节点的调试
	 */
	public void NodeDebug(){
		Node node;
		System.out.println("(x,y,alpha,beta,cost)");
		for(int i=0;i<xsize;i++){
			for(int j=0;j<ysize;j++){
				node=Lattic(i,j);
				System.out.println("("+node.x+","+node.y+","+node.alpha+","+node.beta+","+node.cost+","+node.bestCost+")");
			}
		}
	}
	/**
	 * 路径的调试
	 */
	public void CPathDebug(){
		System.out.println("{(lnode.x,lnode.y)->(rnode.x,rnode.y)}:cost");
		for(int cur=0;cur<xsize;cur++){//路径
			for(int j=0;j<ysize;j++){
				ArrayList<CPath>    lpathAL=Lattic(cur,j).lpath;
				for(CPath path : lpathAL){
					System.out.println("{("+path.lnode.x+","+path.lnode.y+")->("+path.rnode.x+","+path.rnode.y+")}:"+path.cost);
				}
			}
		}
	}
	/**
	 * 期望的调试
	 */
	public void ExpectationDebug(){
		System.out.println();
		System.out.println("ExpectationDebug():");
		for(int i=0;i<expected.size();i++){
			System.out.println("expected["+i+"]:"+expected.get(i));
		}
	}
	
}
