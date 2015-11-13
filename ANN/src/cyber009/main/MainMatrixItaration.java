/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.main;

import cyber009.ann.ANN;
import cyber009.lib.Variable;
import java.util.Random;

/**
 *
 * @author pavel
 */
public class MainMatrixItaration {
    
    
    
    public static void main(String[] args) {
        Random r = new Random(System.currentTimeMillis());
        Variable v = new Variable();
        long timeStart=0, timeEnd=0;
        v.N = 5;
        for(int id =10; id<=4000; id+=100) {
            v.D = id;
            v.threshold = 13e-23;
            cyber009.function.LinearFunction func= new cyber009.function.LinearFunction(v.N);
            v.X = new double[v.D][];
            v.TARGET = new double[v.D];
            for(int d=0; d<v.D; d++) {
                v.X[d] = new double[v.N+1];
                v.X[d][0] = 1.0;
                for(int n=1; n<=v.N; n++) {
                    v.X[d][n] = r.nextGaussian();
                }
                v.TARGET[d] =func.linearFunction(v.X[d]);
            }
            v.WEIGHT = new double[v.N+1];        
            ANN ann = new ANN(v, 0.014013);

            //func.showCoefficients();
            
            System.out.println("###########################\nins:"+v.D);
            ann.weightReset();
            System.out.println("--------gradientDescent---------------");
            timeStart = System.currentTimeMillis();
            ann.gradientDescent(90L);
            timeEnd = System.currentTimeMillis();
            //v.showWEIGHT();
            v.showResult();
            System.out.println("time ("+timeStart+"-"+timeEnd+"):"+(timeEnd-timeStart)+ "ms");
            ann.weightReset();
            System.out.println("----------Matrix-------------");
            timeStart = System.currentTimeMillis();
            ann.weightFindMatrix();
            timeEnd = System.currentTimeMillis();
           // v.showWEIGHT();
            v.showResult();
            System.out.println("time ("+timeStart+"-"+timeEnd+"):"+(timeEnd-timeStart)+ "ms");
        }
    }
    
}
