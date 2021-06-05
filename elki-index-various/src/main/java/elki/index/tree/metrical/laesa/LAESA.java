/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2021
 * ELKI Development Team
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package elki.index.tree.metrical.laesa;

import java.util.Random;

import elki.data.type.TypeInformation;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.DoubleDataStore;
import elki.database.datastore.WritableDoubleDataStore;
import elki.database.ids.*;
import elki.database.query.distance.DistanceQuery;
import elki.database.query.knn.KNNSearcher;
import elki.database.query.range.RangeSearcher;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.index.IndexFactory;
import elki.index.KNNIndex;
import elki.index.RangeIndex;
import elki.logging.Logging;
import elki.logging.LoggingUtil;
import elki.logging.progress.FiniteProgress;
import elki.utilities.documentation.Reference;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import elki.utilities.optionhandling.parameters.RandomParameter;
import elki.utilities.random.RandomFactory;

/**
 * Linear Approximating and Eliminating Search Algorithm
 * <p>
 * Reference:
 * <p>
 * L. Micó, J. Oncina, E. Vidal<br>
 * A new version of the nearest-neighbour approximating and eliminating search
 * algorithm (AESA) with linear preprocessing time and memory requirements<br>
 * Pattern Recognit. Lett. 15(1)
 * 
 * @author Erich Schubert
 * 
 * @param <O> Object type
 */
@Reference(authors = "L. Micó, J. Oncina, E. Vidal", //
    title = "A new version of the nearest-neighbour approximating and eliminating search  algorithm (AESA) with linear preprocessing time and memory requirements", //
    booktitle = "Pattern Recognit. Lett. 15(1)", //
    url = "https://doi.org/10.1016/0167-8655(94)90095-7", //
    bibkey = "DBLP:journals/prl/MicoOV94")
public class LAESA<O> implements RangeIndex<O>, KNNIndex<O> {
  /**
   * Class logger.
   */
  private static final Logging LOG = Logging.getLogger(LAESA.class);

  /**
   * Distance function
   */
  Distance<? super O> distance;

  /**
   * Distance query, bound to the relation
   */
  DistanceQuery<? super O> distq;

  /**
   * Number of reference points
   */
  int m;

  /**
   * Random generator
   */
  RandomFactory rng;

  /**
   * Relation indexed.
   */
  Relation<O> relation;

  /**
   * Reference points
   */
  ArrayModifiableDBIDs refp;

  /**
   * Data storage for precomputed distances to reference points.
   */
  DoubleDataStore[] dists;

  /**
   * Constructor.
   *
   * @param distance Distance function
   * @param m Number of reference points
   * @param rng Random generator
   */
  public LAESA(Relation<O> relation, Distance<? super O> distance, int m, RandomFactory rng) {
    this.relation = relation;
    this.distance = distance;
    this.distq = distance.instantiate(relation);
    this.m = m;
    this.rng = rng;
  }

  @Override
  public void initialize() {
    DBIDs ids = relation.getDBIDs();
    ArrayDBIDs aids = DBIDUtil.ensureArray(ids);
    Random rand = rng.getSingleThreadedRandom();
    int best = rand.nextInt(aids.size()); // First is chosen randomly
    DBIDVar cur = DBIDUtil.newVar();
    // Store chosen reference points
    refp = DBIDUtil.newArray(m);
    // Fast lookup for existing reference points
    HashSetModifiableDBIDs chosen = DBIDUtil.newHashSet(m);
    WritableDoubleDataStore a = DataStoreUtil.makeDoubleStorage(ids, DataStoreFactory.HINT_TEMP | DataStoreFactory.HINT_HOT, 0);
    WritableDoubleDataStore ds;
    FiniteProgress rprog = LOG.isVerbose() ? new FiniteProgress("Computing distances to reference points", m, LOG) : null;
    FiniteProgress distances = LOG.isVerbose() ? new FiniteProgress("Distance computations", ids.size() * m - m * (m + 1) / 2, LOG) : null;
    for(int i = 0; i < m; i++) {
      refp.add(aids.assignVar(best, cur));
      chosen.add(cur);
      best = 0;
      double besta = 0;
      ds = DataStoreUtil.makeDoubleStorage(ids, DataStoreFactory.HINT_HOT);
      for(DBIDArrayIter iter = aids.iter(); iter.valid(); iter.advance()) {
        if(chosen.contains(iter)) {
          if(DBIDUtil.equal(iter, cur)) {
            ds.putDouble(iter, 0);
            continue;
          }
          for(DBIDArrayIter l = refp.iter(); l.valid(); l.advance()) {
            if(DBIDUtil.equal(iter, l)) {
              ds.putDouble(iter, dists[l.getOffset()].doubleValue(iter));
              break;
            }
          }
          continue;
        }
        final double d = distq.distance(cur, iter);
        ds.putDouble(iter, d);
        // For choosing the next reference point:
        double ap = a.doubleValue(iter) + d;
        a.putDouble(iter, ap);
        if(ap > besta) {
          besta = ap;
          best = iter.getOffset();
        }
        LOG.incrementProcessed(distances);
      }
      dists[i] = ds;
      LOG.incrementProcessed(rprog);
    }
    LOG.ensureCompleted(rprog);
  }

  @Override
  public KNNSearcher<O> kNNByObject(DistanceQuery<O> distanceQuery, int maxk, int flags) {
    // TODO Auto-generated method stub
    return null;
  }

  @Override
  public RangeSearcher<O> rangeByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
    // TODO Auto-generated method stub
    return null;
  }

  /**
   * Index factory.
   * 
   * @author Erich Schubert
   *
   * @param <O> Object type
   */
  public static class Factory<O> implements IndexFactory<O> {
    /**
     * Distance function
     */
    Distance<? super O> distance;

    /**
     * Number of reference points
     */
    int m = 10;

    /**
     * Random generator
     */
    RandomFactory rng;

    /**
     * Constructor.
     *
     * @param distance Distance function
     * @param m Number of reference points
     * @param rng Random generator
     */
    public Factory(Distance<? super O> distance, int m, RandomFactory rng) {
      this.distance = distance;
      this.m = m;
      this.rng = rng;
    }

    @Override
    public LAESA<O> instantiate(Relation<O> relation) {
      return new LAESA<O>(relation, distance, m, rng);
    }

    @Override
    public TypeInformation getInputTypeRestriction() {
      return distance.getInputTypeRestriction();
    }

    /**
     * Parameterization class.
     *
     * @author Erich Schubert
     * 
     * @param <O> object type
     */
    public static class Par<O> implements Parameterizer {
      /**
       * Distance function to use in the index
       */
      public static final OptionID DISTANCE_FUNCTION_ID = new OptionID("laesa.distance", "Distance function to determine the distance between objects.");

      /**
       * Number of reference points to choose.
       */
      public static final OptionID M_ID = new OptionID("laesa.m", "Number of reference points to use.");

      /**
       * Random generator to use
       */
      public static final OptionID SEED_ID = new OptionID("laesa.seed", "Random generator seed to use.");

      /**
       * Distance function
       */
      Distance<? super O> distance;

      /**
       * Number of reference points
       */
      int m;

      /**
       * Random generator
       */
      RandomFactory rng;

      @Override
      public void configure(Parameterization config) {
        new ObjectParameter<Distance<? super O>>(DISTANCE_FUNCTION_ID, Distance.class) //
            .grab(config, x -> {
              distance = x;
              if(!distance.isMetric()) {
                LoggingUtil.warning("LAESA requires a metric to be exact.");
              }
            });
        new IntParameter(M_ID, 10) //
            .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
            .grab(config, x -> m = x);
        new RandomParameter(SEED_ID) //
            .grab(config, x -> rng = x);
      }

      @Override
      public Factory<O> make() {
        return new Factory<O>(distance, m, rng);
      }
    }
  }
}
