/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.ann;

import cyber009.lib.Variable;
import java.util.Random;
import weka.core.matrix.Matrix;

/**
 *
 * @author pavel
 */
public class ANN {
    
    public Variable v;
    public double learnRate;
    //public Function.LinearFunction func;
    
    public ANN(Variable v, double learnRate) {
        this.v = v;
        this.learnRate = learnRate;
        //this.func = new LinearFunction(v.N);
    }
    
    public double sign(double v) {
        if(v>=0) {
            return 1.0;
        }
        return -1.0;
    }
    
    public double threshold(double a) {
        if(a>v.threshold) {
            return 1.0;
        }
        return 0.0;
    }
    
    public double logistic(double a) {
        return 1.0000001/(1.0000001 + Math.exp(-a));
    }
    
    public double tanH(double a) {
        double v =Math.tanh(a);
        if(v==Double.NaN) {
            return 0.0;
        }
        return v;        
    }
    
    
    public double calOutput(int dataSetindex, int mode) {
        double sum =0.0;
        for(int i = 0; i<=v.N; i++) {
            sum += (v.X[dataSetindex][i] * v.WEIGHT[i]);
        }
        if(mode==1) {
            return threshold(sum);
        } else if(mode==2){
            return logistic((sum));            
        } else if(mode==3) {
            return tanH(sum);
        }
        return sum;
    }
    
    public void trainingFunction() {
        
    }
    
    public long gradientDescent(long itaration) {
        return gradientDescent(itaration, 0, v.D);
    }
    
    public long gradientDescent(long itaration, int mode, int D) {
        Random rand = new Random(System.currentTimeMillis());
        double []deltaW = new double[v.N+1];
        double []tempW = new double[v.N+1];
        double totalError = 0.0, temp;
        long count = 0;
        int i, d;
        boolean changeWeight = false;
        boolean unMatch = false;
        // small random value set to weight
        for(i=0; i<=v.N; i++) {
            v.WEIGHT[i] = Math.abs(rand.nextDouble());
            v.WEIGHT[i] = (double) v.WEIGHT[i] - Math.floor(v.WEIGHT[i]);
        }        
        do {
            for(i= 0; i<=v.N; i++) {
                tempW[i] = v.WEIGHT[i];
            }
            for(i=0; i<=v.N; i++) {
                deltaW[i] = 0;
                for(d=0; d<D; d++) {
                    deltaW[i] += learnRate * 
                            ( v.TARGET[d] - (calOutput(d, mode))) *
                            v.X[d][i];
                }
            }
            for(i=0; i<=v.N; i++) {
                v.WEIGHT[i] += deltaW[i];
            }
            
            unMatch = false;
            for(d=0; d<D; d++) {
                if((calOutput(d, 0))!= v.TARGET[d]) {
                    unMatch = true;
                    break;
                }
            }
            
            changeWeight = false;
            for(i=0; i<=v.N; i++) {
                if(tempW[i] != v.WEIGHT[i]) {
                    changeWeight = true;
                    break;
                }
            }
            
            count++;
//            if(count%100000 == 0) {
//                System.out.println("-----------Count: "+ count+"--------------");
//                //v.showWEIGHT();
//                //6v.showTable(0, 6);
//                System.out.println("------------------------------------------");
//            }
//            if((count ==itaration) || (changeWeight==false) || (unMatch == false)) {
//                System.out.println("End:"+count+ " (WeightChange:"+changeWeight+") (target!=output:"+unMatch+")");
//            }
            if((count >itaration)) { // some proble in NaN double data
                break;
            }
        } while((changeWeight==true) && (unMatch == true));
        //Weight = {3.2198905626106613E-4, 2.2335823691292756E-4, 3.59674533175504E-4 }
//        System.out.println("---Final----Count: "+ count+"--------------");
//        v.showWEIGHT();
//        //v.showTable();
//        System.out.println("------------------------------------------");
        return count;
    }
    
    public void weightFindMatrix() {
        Matrix X = new Matrix(v.X);
        Matrix Y = new Matrix(v.D, 1);
        Matrix W = new Matrix(v.N, 1);
        
        for(int d = 0; d<v.D; d++) {
            Y.set(d, 0, v.TARGET[d]);
        }
        
        for(int n = 0; n<v.N; n++) {
            W.set(n, 0, 0.0);
            //W.set(n, 0, v.WEIGHT[n]);
        }
        
        Matrix temp = X.transpose().times(X);
//        System.out.println(temp.toString());
        temp = temp.inverse().times(X.transpose());
//        System.out.println(temp.toString());
        temp = temp.times(Y);
        //System.out.println(temp.toString());
        W = temp;
        for(int n=0; n<=v.N; n++) {
            v.WEIGHT[n] = W.get(n, 0);
        }
        //System.out.println(YI.toString());
        
    }
    
    
    public void weightReset() {
        for(int i=0; i<v.N; i++) {
            v.WEIGHT[i] = 0.0;
        }
    }
    
}
