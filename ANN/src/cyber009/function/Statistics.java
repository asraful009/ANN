
package cyber009.function;

import cyber009.lib.Variable;
import java.util.HashMap;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.matrix.Matrix;
import weka.estimators.MultivariateGaussianEstimator;

/**
 *
 * @author pavel
 */
public class Statistics {
    
    public HashMap<Double, Matrix> mu;
    public HashMap<Double, Matrix> sigma;
    public Variable V;
    public Statistics(Variable V) {
        this.V = V;
        mu = new HashMap<>();
        sigma = new HashMap<>();
    }
    
    public static double getBiMultivariant(Instances ins, Matrix X) {
        int k = ins.numAttributes()-1;        
        double[][] means = new double[k][1];        
        double [][] sd = new double[k][k];
        for(int i=0; i<k; i++) {
            means[i][0] = getMeans(ins, i);
        }        
        for(int r=0; r<k; r++) {
            for(int c=0; c<k; c++) {
                sd[r][c] = 0.0;
                for (Instance data : ins) {
                    sd[r][c] +=( (data.value(r)- means[r][0])*
                            (data.value(c)- means[c][0]));
//                    System.out.println("("+data.value(r)+"-"+ means[r][0]+") * ("+
//                            data.value(c)+"-"+ means[c][0]+") = "+sd[r][c]);                    
                }
                sd[r][c] = (sd[r][c]/
                              (double)((ins.size()>1?ins.size()-1:1.0)));                
            }
        }
        //X = new Matrix(means);
        Matrix mu = new Matrix(means);
        Matrix sigma = new Matrix(sd);
        Matrix xmu = X.minus(mu);
        double det = sigma.det();
        double constance = (1.0)/ (Math.sqrt(
                                    Math.pow(2.00*Math.PI, k)*
                                    det));
        Matrix eM = (xmu.transpose().times(sigma.inverse().times(xmu)));
        double exp = Math.exp((-1.0/2.0)*eM.get(0, 0));
        double ret = constance*exp;
        ret= (ret>1.0?1.0:ret);
        System.out.println(ret);
        return ret;
    }
    
    public static double getMultivariantProbability(Instances ins, Matrix X) {
        double[][] dataset = new double[ins.size()][ins.numAttributes()-1];
        double[] means_1 = new double[ins.numAttributes()-1];
        MultivariateGaussianEstimator mv = new MultivariateGaussianEstimator();
        double[][] newData = X.transpose().getArrayCopy();//new double[] {  0.2, 0.1, 0.43, 0.29, 0.0 };
        int i=0, j;
        for (Instance data : ins) {
            for(j=0; j<ins.numAttributes()-1; j++) {
                dataset[i][j] = data.value(j);
            }
            i++;
        }
        double [] w = new double[ins.size()];
        int t = ins.size();
        double d = (double)1.0/t;
        for(i=0; i<t; i++) {
            w[i] = d;
        }
        mv.estimate(dataset,w);
        //Matrix mv_means = new Matrix(mv.getMean(), 1);
        //Matrix mv_cov = new Matrix(mv.getCovariance());    
        //System.out.println(mv_means);
       // System.out.println(mv_cov);
        double ret = mv.getProbability(newData[0]);
        System.out.println(ret);
        return ret;
    }
    
    public void calMultiVariantMuSigma(double target) {
        int k = V.N;
        int count = 0;
        double[][] means = new double[k][1];
        double [][] sd = new double[k][k];
        for(int i=0; i<k; i++) {
            means[i][0] = getMeans(target, i);
        }      
        for(int r=0; r<k; r++) {
            for(int c=0; c<k; c++) {
                sd[r][c] = 0.0;
                count = 0;
                for (int d=0; d<V.D; d++) {
                    if(V.TARGET[d]==target && V.LABEL[d]==true) {
                        sd[r][c] +=( (V.X[d][r]- means[r][0])*
                                (V.X[d][r]- means[c][0]));
                        count++;
//                    System.out.println("("+data.value(r)+"-"+ means[r][0]+") * ("+
//                            data.value(c)+"-"+ means[c][0]+") = "+sd[r][c]);                    
                    }
                }
                sd[r][c] = (sd[r][c]/
                       (double)((count>0?count:1.0)));                
                
            }
        }
        mu.put(target, new Matrix(means));
        sigma.put(target, new Matrix(sd));
    }
    
    public double posteriorDistribution(double target, Matrix val) {
        
        int k = val.getRowDimension();
        System.out.println("posterior Distribution "+ k +"\n"+ val.toString());
        Matrix xmu = val.minus(mu.get(target));
        double det = sigma.get(target).det();
        double constance = (1.0)/ (Math.sqrt(
                                    Math.pow(2.00*Math.PI, k)*
                                    det));
        Matrix eM = (xmu.transpose().times(sigma.get(target).inverse().times(xmu)));
        double exp = Math.exp((-1.0/2.0)*eM.get(0, 0));
        double ret = constance*exp;
        ret= (ret>1.0?1.0:ret);
        System.out.println(ret);
        return ret;
    }
    
    public static double getMeans(Instances ins, int index) {
        return ins.meanOrMode(index);
    }
    
    
    public double getMeans(double target, int index) {
        double sum = 0.0;
        int count = 0;
        for(int d=0; d<V.D; d++) {
            if(V.TARGET[d]== target && V.LABEL[d]==true) {
                sum +=V.X[d][index];
                count++;
            }
        }
        if(count==0) {
            return 0.0;
        }
        sum /=(double) count;
        return sum;
    }
}
 