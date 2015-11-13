
package cyber009.main;

import cyber009.function.Statistics;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.matrix.Matrix;

/**
 *
 * @author pavel
 */
public class MainLMV {
    
    public static void main(String[] args) {
        InputStream is = null;
        try {
            is = new FileInputStream("data/mark.arff");
            DataSource source = new DataSource(is);
            Instances instances = source.getDataSet();
            // Make the last attribute be the class
            instances.setClassIndex(instances.numAttributes() - 1);
            double [][] test = new double[][]{{ 0.05}, {0.01}, {0.43}, {0.29}, {0.56}};
            Statistics.getBiMultivariant(instances, new Matrix(test));            
            Statistics.getMultivariantProbability(instances, new Matrix(test));
            
            
        } catch (FileNotFoundException ex) {
            Logger.getLogger(MainLMV.class.getName()).log(Level.SEVERE, null, ex);
        } catch (Exception ex) {
            Logger.getLogger(MainLMV.class.getName()).log(Level.SEVERE, null, ex);
        } finally {
            try {
                is.close();
            } catch (IOException ex) {
                Logger.getLogger(MainLMV.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }
}
