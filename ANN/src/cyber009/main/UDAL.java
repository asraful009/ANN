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
public class UDAL {
    
    Random r;
    Variable v;
    ANN ann;
    cyber009.function.LinearFunction func;
    
    public UDAL(double learnRate) {
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
        v.WEIGHT = new double[v.N+1];
        v.LABEL = new boolean[D+1];
        for(int d=0; d<v.D; d++) {
            v.X[d] = new double[v.N+1];
            v.X[d][0] = 1.0;
            for(int n=1; n<=v.N; n++) {
                v.X[d][n] = r.nextGaussian();
            }            
        }
    }
    
    public Instances activeLearning(int s, int D) {
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
        Instances dataSet = new Instances("Syn Data "+s+ "-"+D, (ArrayList<Attribute>) atts, v.D);        
        Instance ins = new DenseInstance(v.N+2);
        for(int d=s; d<(s+D); d++) {
            v.TARGET[d] =func.syntacticFunction(v.X[d], v.threshold);
            v.LABEL[d]=true;
            for(int n = 0; n<v.N; n++) {
                ins.setValue(atts.get(n), v.X[d][n]);
            }
            ins.setValue(atts.get(v.N+1), v.TARGET[d]);
            dataSet.add(ins);
        }
        return dataSet;
    }
    
    public void showData() {
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
            Instances dataSet = new Instances("Syn Data", (ArrayList<Attribute>) atts, v.D);

            for(int d= 0; d<v.D; d++) {
                Instance ins = new DenseInstance(v.N+2);
                for(int i=0; i<=v.N; i++) {
                    ins.setValue(atts.get(i), v.X[d][i]);
                }
                ins.setValue(atts.get(v.N+1), v.TARGET[d]);
                dataSet.add(ins);
            }
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
        
        UDAL udal = new UDAL(0.014013);
        Statistics statis = new Statistics();
        long timeStart=0, timeEnd=0;
        for(int f=2; f<=2; f++) {
            udal.initUDAL(4, 4000);
            Instances ins = udal.activeLearning(0, 40);
            udal.ann.weightReset();
            timeStart = System.currentTimeMillis();
            udal.ann.gradientDescent(10000L, 3, 40);
            statis.calMultiVariantMuSigma(ins);
            System.out.println(statis.mu);
            System.out.println(statis.sigma);
            timeEnd = System.currentTimeMillis();
            System.out.println("feature #:"+udal.v.N+" time:("+ (timeEnd - timeStart) +")");
            udal.v.showResult();
            
            
        }
    }
}
