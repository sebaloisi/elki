package de.lmu.ifi.dbs.wrapper;

import de.lmu.ifi.dbs.database.Database;
import de.lmu.ifi.dbs.database.RTreeDatabase;
import de.lmu.ifi.dbs.parser.Parser;
import de.lmu.ifi.dbs.parser.StandardLabelParser;
import de.lmu.ifi.dbs.distance.DistanceFunction;
import de.lmu.ifi.dbs.distance.EuklideanDistanceFunction;
import de.lmu.ifi.dbs.utilities.QueryResult;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;
import java.util.List;

/**
 *
 */
public class Test {
  public static void main(String[] args) {
    try {
      File file1 = new File("test.txt");
      InputStream in1 = new FileInputStream(file1);
      Parser parser1 = new StandardLabelParser();
      String[] param1 = {"-database", "de.lmu.ifi.dbs.database.RTreeDatabase",
                         "-" + RTreeDatabase.FILE_NAME_P, "elki.idx"};
      parser1.setParameters(param1);
      Database db1 = parser1.parse(in1);
      System.out.println(db1);

      File file2 = new File("test.txt");
      InputStream in2 = new FileInputStream(file2);
      Parser parser2 = new StandardLabelParser();
      String[] param2 = {"-database", "de.lmu.ifi.dbs.database.RTreeDatabase"};
      parser2.setParameters(param2);
      Database db2 = parser2.parse(in2);
      System.out.println(db2);

      DistanceFunction distFunction = new EuklideanDistanceFunction();
      List<QueryResult> r1 = db1.kNNQuery(new Integer(Integer.MIN_VALUE + 300), 10, distFunction);
      System.out.println("r1 "+r1);

      List<QueryResult> r2 = db2.kNNQuery(new Integer(Integer.MIN_VALUE +300), 10, distFunction);
      System.out.println("r2 "+r2);

//      for (int i = 0; i < 450; i++) {
//        MetricalObject o = db1.get(new Integer(Integer.MIN_VALUE + i));
//        db1.delete(o);
//        System.out.println(db1);
//      }
    }
    catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }
}
