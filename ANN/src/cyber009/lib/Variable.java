/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.lib;

/**
 *
 * @author pavel
 */
public class Variable {
    public int N;
    public int D;
    public double threshold;
    public double [][] X;
    public double [] WEIGHT;
    public double [] TARGET;
    public boolean [] LABEL;
    
    public Variable() {
        
    }
    
    public Variable(int N) {
        this.N = N;
        X = new double[N+1][];
        WEIGHT = new double[N+1];        
        WEIGHT[0] = 1.0;
    }
    
    public Variable(int N, int D) {
        this.N = N;
        this.D = D;
        X = new double[N+1][];
        WEIGHT = new double[N+1];
        TARGET = new double[D];
        LABEL = new boolean[D+1];
        for(int i=0; i<=N; i++) {
            X[i] = new double[D];
        }
        WEIGHT[0] = 1.0;
    }
    
    public double threshold(double v) {
        if(v>threshold) {
            return 1.0;
        }
        return 0.0;
    }
    
    public double logistic(double a) {
        return 1.0000001/(1.0000001 + (double)Math.exp(-a));
    }
    
    public void showX() {
        
        for(int d=0; d<D; d++) {
            System.out.print("X = {");
            for(int i=0; i<N; i++) {
                System.out.print(X[d][i]+ ", ");
            }
            System.out.println(X[d][N]+ " }");
        }
        
    }
    
    public void showAll() {
        System.out.println("x0, x1, x2, class");
        for(int d=0; d<D; d++) {
            System.out.print("");
            for(int i=0; i<N; i++) {
                System.out.print(X[d][i]+ ", ");
            }
            System.out.println(X[d][N]+ ", "+TARGET[d]+";");
        }
    }
    
    public void showWEIGHT() {
        System.out.print("Weight = {");
        for(int i=0; i<N; i++) {
            System.out.print(WEIGHT[i]+ ", ");
        }
        System.out.println(WEIGHT[N]+ " }");
    }
    
    public void showTable() {
        showTable(0, D);
    }
    
    public void showTable(int s, int e) {
        double out = 0.0, error;
        int count_error = 0;
        String str = "";
        for(int i=1; i<=N; i++) {
            System.out.print("X["+i+"]\t|");
        }
        System.out.println("Target\t|Output\t|Error|");
        for(int d=s; d<e; d++) {
            out = X[d][0]*WEIGHT[0];
            str = X[d][0]+"*"+WEIGHT[0];
            //error = (TARGET[d] - out) * (TARGET[d] - out);
            for(int i=1; i<=N; i++) {
                System.out.print(X[d][i]+"\t|");
                out += X[d][i]*WEIGHT[i];
                str += " + "+X[d][i]+"*"+WEIGHT[i];
                //error += (TARGET[d] - out) * (TARGET[d] - out);
            }
            //out= logistic(out);
            out= threshold(out);
            if(TARGET[d] != out) {
                count_error++;
            }
            error = Math.sqrt(((TARGET[d] - out)*(TARGET[d] - out))/2.0000000001);
            System.out.println(
                    TARGET[d]+"\t|"
                    +out+"\t|"+error+" \t|"+str);
        }
        System.out.println("total wrong:"+ count_error+"/"+D);
    }
    
    public void showResult() {
        double out = 0.0, error = 0.0 ;
        int count_error = 0;
        String str = "";
//        for(int i=1; i<=N; i++) {
//            System.out.print("X["+i+"]\t|");
//        }
        //System.out.println("Target\t|Output\t|Error|");
        for(int d=0; d<D; d++) {
            out = X[d][0]*WEIGHT[0];
            str = X[d][0]+"*"+WEIGHT[0];
            for(int i=1; i<=N; i++) {                
                out += X[d][i]*WEIGHT[i];
                str += " + "+X[d][i]+"*"+WEIGHT[i];                
            }
            //out= logistic(out);
            out= (out);
            
            
            error += Math.pow((TARGET[d] - out), 2.00000);
//            System.out.println(
//                    TARGET[d]+"\t|"
//                    +out);//+"\t|"+error+" \t|"+str);
        }
        double rmsError = Math.sqrt(error)/(double)D*100.0;
        System.out.println("rms Error:"+rmsError+" %");
    }
    
    public void showSummary() { 
//        int count_error = 0;
//        for(int d=0; d<D; d++) {
//            if(TARGET[d] == )
//        }
    }
    
}
