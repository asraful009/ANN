/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.function;

import java.util.Random;

/**
 *
 * @author pavel
 */
public class LinearFunction {
    
    Random r = new Random(System.currentTimeMillis());
    double [] coefficients;
    int N;
    
    public LinearFunction(int N) {
        this.N = N;
        coefficients = new double[N+1];
        for(int n=0; n<=N; n++) {
            coefficients[n] = r.nextGaussian();
        }
    }
        
    public void showCoefficients() {
        System.out.print("coeffi = {");
        for(int i=0; i<coefficients.length-1; i++) {
            System.out.print(""+coefficients[i]+", ");
        }
        System.out.println(coefficients[coefficients.length-1]+" }");
    }
    
    public double linearFunction(double [] x) {
        double y = r.nextGaussian();
        for(int i=0; i<x.length; i++) {
            y += x[i]*coefficients[i];
        }
        return y;
    }
    
    public static double linearCosFunction(double x) {
        return Math.cos(x);
    }
    
    public static double LogicAND(double [] X) {
        for(int i=1; i<X.length; i++) {
            if(X[i] == 0.0) {
                return 0.0;
            }
        }
        return 1.0;
    }
    
    public static double LogicOR(double [] X) {
        for(int i=1; i<X.length; i++) {
            if(X[i] == 1.0) {
                return 1.0;
            }
        }
        return 0.0;
    }
    
    public static double LogicXOR(double [] X) {
        int count =0;
        for(int i=1; i<X.length; i++) {
            if(X[i] == 1.0) {
                count++;
            }
        }
        
        if(count %2 ==0)
            return 0.0;
        return 1.0;
    }
    
    public double syntacticFunction(double []X, double th) {
        double ret = r.nextGaussian();
        for(int i=0; i<X.length; i++) {
            ret += X[i]*coefficients[i];
        }
        if(ret > 0.0) {
            return 1.0;
        }
        return 0.0;
    }
    
}
