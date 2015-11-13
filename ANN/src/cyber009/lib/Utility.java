/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package cyber009.lib;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author pavel
 */
public class Utility {
    
    public static void readDataSet(String Path, Variable v) {
        
        
    }
    
    public static void writeCSVDataSet(String path, Variable v) {
        PrintWriter out;
        try {
            out = new PrintWriter(path, "UTF-8");
            out.println("x0, x1, x2, class");
            for(int d=0; d<v.D; d++) {
                out.print("");
                for(int i=0; i<v.N; i++) {
                    out.print(v.X[d][i]+ ", ");
                }
                out.println(v.X[d][v.N]+ ", "+v.TARGET[d]+";");
            }
            out.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(Utility.class.getName()).log(Level.SEVERE, null, ex);
        } catch (UnsupportedEncodingException ex) {
            Logger.getLogger(Utility.class.getName()).log(Level.SEVERE, null, ex);
        }
        
    }
    
}
