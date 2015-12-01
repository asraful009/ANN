/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.main;

import cyber009.ann.ANN;
import cyber009.function.Statistics;
import cyber009.lib.Variable;
import java.awt.BorderLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.swing.JFrame;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.visualize.PlotData2D;
import weka.gui.visualize.VisualizePanel;

/**
 *
 * @author pavel
 */
public class UDALNeuralNetwork {
    
    Random r;
    Variable v;
    ANN ann;
    cyber009.function.LinearFunction func;
    Instances dataSet = null;
    public UDALNeuralNetwork(double learnRate) {
        r = new Random(System.currentTimeMillis());
        v = new Variable();
        ann = new ANN(v, learnRate);//0.014013);
    }
    
    public void initUDAL(int f, int D) {
        v.N = f;
        v.D = D;
        v.threshold = 0.0;
        func = new cyber009.function.LinearFunction(v.N);
        v.X = new double[v.D][];
        v.TARGET = new double[v.D];
        v.LABEL = new boolean[D+1];
        for(int d=0; d<v.D; d++) {
            v.X[d] = new double[v.N];
            for(int n=1; n<=v.N; n++) {
                v.X[d][n] = r.nextGaussian();
            }            
        }
    }
    
    public void activeLearning(int s, int D) {
        for(int d=s; d<(s+D); d++) {
            v.TARGET[d] =func.syntacticFunction(v.X[d], v.threshold);
            if(isClassHave(v.TARGET[d]) == false) {
                v.CLASSES.add(v.TARGET[d]);
            }
            v.N_DATA_IN_CLASS.put(v.TARGET[d] , v.N_DATA_IN_CLASS.getOrDefault(v.TARGET[d], 0)+1);
            v.LABEL[d]=true;            
        }       
    }
    
    public boolean isClassHave(Double CLASS) {
        for (Double target : v.CLASSES ) {
            if(target.equals(CLASS)) {
                return true;
            }
        }
        return false;
    }
    
    public void arraytoInstances() {
        List<Attribute> atts = new ArrayList<>();
        Attribute [] att = new Attribute[v.N+2];
        for(int i=0; i<=v.N; i++) {
            att[i] = new Attribute("X"+i);
            atts.add(att[i]);
        }
        List<String> classValus = new ArrayList<>();
        classValus.add("1.0");
        classValus.add("0.0");
        att[v.N+1] = new Attribute("class", classValus);
        atts.add(att[v.N+1]);
        dataSet = new Instances("Syn Data", (ArrayList<Attribute>) atts, v.D);
        dataSet.setClass(att[v.N+1]);
        for(int d= 0; d<v.D; d++) {
            Instance ins = new DenseInstance(v.N+2);
            for(int i=0; i<=v.N; i++) {
                ins.setValue(atts.get(i), v.X[d][i]);
            }
            ins.setValue(atts.get(v.N+1), v.TARGET[d]);
            dataSet.add(ins);
        }
        
    }
    
    public void showData() {
        //System.out.println(dataSet);
        PlotData2D p2D = new PlotData2D(dataSet);
        p2D.setPlotName("Syn data");
        VisualizePanel vp = new VisualizePanel();
        vp.setName("Show Data");
        try {
            vp.addPlot(p2D);

            JFrame frame = new JFrame("Show Data");
            frame.setSize(600, 600);
            frame.setVisible(true);
            frame.getContentPane().setLayout(new BorderLayout());
            frame.getContentPane().add(vp, BorderLayout.CENTER);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setVisible(true);
            func.showCoefficients();
        } catch (Exception ex) {
            Logger.getLogger(MainSyntacticData.class.getName()).log(Level.SEVERE, null, ex);
        }

    }
    
    public static void main(String[] args) {        
        UDALNeuralNetwork udal = new UDALNeuralNetwork(0.014013);
        Statistics statis = new Statistics(udal.v);
        long timeStart=0, timeEnd=0;
        for(int f=2; f<=2; f++) {
            udal.initUDAL(4, 5000);
            udal.activeLearning(0, 5000);
            udal.arraytoInstances();
            udal.ann.weightReset();
            timeStart = System.currentTimeMillis();
            MultilayerPerceptron wekaNN = new MultilayerPerceptron();
            wekaNN.setAutoBuild(true);
            //wekaNN.setGUI(true);
            try {
                wekaNN.buildClassifier(udal.dataSet);
                Evaluation eval = new Evaluation(udal.dataSet);
                System.out.println(wekaNN.toString());
                eval.crossValidateModel(wekaNN, udal.dataSet, 4999, new Random(System.currentTimeMillis()));
                System.out.println(wekaNN.toString());
                System.out.println(eval.toClassDetailsString());
                
//            udal.ann.gradientDescent(10000L, 3, 100);
//            for (Double target : udal.v.CLASSES) {
//                statis.calMVMuSigma(target);
//                System.out.println(udal.v.N_DATA_IN_CLASS.get(target));
//                System.out.println(statis.mu.get(target));
//                System.out.println(statis.sigma.get(target));
//            }
//            for(int d=0; d<udal.v.D; d++) {
//                if(udal.v.LABEL[d] == false) {
//                    double [][] val = new double[udal.v.N-1][1];
//                    for(int n=1; n<udal.v.N; n++) {
//                        val[n-1][0] = udal.v.X[d][n];
////                        System.out.print(udal.v.X[d][n] + "   ");
////                        System.out.println(val[n-1][0]);
//                    }
//                    Matrix mVal = new Matrix(val);
//                    double pp = 0.0D;
//                    for (Double target : udal.v.CLASSES) {
//                        //System.out.println("-----------------------\nClass:"+ target);
//                        pp += statis.posteriorDistribution(target, mVal);
//                        System.out.println("conditional: Entropy: "+ 
//                                statis.conditionalEntropy(target, mVal, d));
//                    }
//                    System.out.print("Sum posterior:"+ pp+ " for "+new Matrix(val).transpose());
//                    
//                }
//            }
//            System.out.println("-----------------------");
//            timeEnd = System.currentTimeMillis();
//            System.out.println("feature #:"+udal.v.N+" time:("+ (timeEnd - timeStart) +")");
//            udal.v.showResult();
//            
            } catch (Exception ex) {
                Logger.getLogger(UDALNeuralNetwork.class.getName()).log(Level.SEVERE, null, ex);
            }
            
        }
    }
}
