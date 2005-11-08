package de.lmu.ifi.dbs.database;

import de.lmu.ifi.dbs.data.MetricalObject;
import de.lmu.ifi.dbs.distance.Distance;
import de.lmu.ifi.dbs.index.metrical.MetricalIndex;
import de.lmu.ifi.dbs.index.metrical.mtree.mkmax.MkMaxTree;

import java.util.List;

/**
 * MDkNNTreeDatabase is a database implementation which is supported by a
 * MDkNNTree index structure.
 *
 * @author Elke Achtert(<a href="mailto:achtert@dbs.ifi.lmu.de">achtert@dbs.ifi.lmu.de</a>)
 */
public class MkMaxTreeDatabase<O extends MetricalObject, D extends Distance> extends MkNNTreeDatabase<O, D> {

  /**
   * Empty constructor, creates a new MDkNNTreeDatabase.
   */
  public MkMaxTreeDatabase() {
    super();
  }

  /**
   * Creates a metrical index object for this database.
   */
  public MetricalIndex<O, D> createMetricalIndex() {
    return new MkMaxTree<O, D>(fileName, pageSize, cacheSize, getDistanceFunction(), k);
  }

  /**
   * Creates a metrical index object for this database.
   *
   * @param objects the objects to be indexed
   */
  public MetricalIndex<O, D> createMetricalIndex(List<O> objects) {
    return new MkMaxTree<O, D>(fileName, pageSize, cacheSize, getDistanceFunction(), k, objects);
  }

  /**
   * @see Database#description()
   */
  public String description() {
    StringBuffer description = new StringBuffer();
    description.append(MkMaxTreeDatabase.class.getName());
    description.append(" holds all the data in a MkMax-Tree index structure.\n");
    description.append(optionHandler.usage("", false));
    return description.toString();
  }

  /**
   * Sets the parameters k to the parameters set by the super-class' method.
   * Parameter k is required.
   *
   * @see de.lmu.ifi.dbs.utilities.optionhandling.Parameterizable#setParameters(String[])
   */
  public String[] setParameters(String[] args) throws IllegalArgumentException {
    return super.setParameters(args);
  }


}
