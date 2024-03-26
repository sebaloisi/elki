/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 * 
 * Copyright (C) 2022
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

package elki.index.tree.metrical.vptree;

import java.util.Random;

import elki.database.ids.ArrayModifiableDBIDs;
import elki.database.ids.DBID;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDMIter;
import elki.database.ids.DBIDRef;
import elki.database.ids.DBIDUtil;
import elki.database.ids.DBIDVar;
import elki.database.ids.DBIDs;
import elki.database.ids.DoubleDBIDListMIter;
import elki.database.ids.ModifiableDoubleDBIDList;
import elki.database.ids.QuickSelectDBIDs;
import elki.database.query.PrioritySearcher;
import elki.database.query.distance.DistanceQuery;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.index.DistancePriorityIndex;
import elki.logging.Logging;
import elki.utilities.datastructures.QuickSelect;
import elki.utilities.documentation.Reference;
import elki.utilities.random.RandomFactory;

/**
 * Vantage Point Tree using k-fold splits
 * <p>
 * In contrast to the standart VP-Tree @see
 * elki.index.tree.metrical.vptree.VPTree,
 * this Variation uses k-fold splits to further partition the the Dataset at
 * every Node
 * into k Child Nodes instead of just two.
 * <p>
 * Based on the ELKI Project Vantage Point Tree Implementation by Robert Gehde
 * and Erich Schubert
 * <p>
 * Reference:
 * <p>
 * P. N. Yianilos<br>
 * Data Structures and Algorithms for Nearest Neighbor Search in General Metric
 * Spaces<br>
 * Proc. ACM/SIGACT-SIAM Symposium on Discrete Algorithms
 * 
 * @author Sebastian Aloisi
 * @since 0.8.1
 * 
 * @param <O> Object Type
 */
@Reference(authors = "P. N. Yianilos", //
        title = "Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces", //
        booktitle = "Proc. ACM/SIGACT-SIAM Symposium on Discrete Algorithms", //
        url = "http://dl.acm.org/citation.cfm?id=313559.313789", //
        bibkey = "DBLP:conf/soda/Yianilos93")

public class VPkTree<O> implements DistancePriorityIndex<O> {
    /**
     * Class logger.
     */
    private static final Logging LOG = Logging.getLogger(VPTree.class);

    /**
     * The representation we are bound to.
     */
    protected final Relation<O> relation;

    /**
     * Distance Function to use
     */
    Distance<? super O> distFunc;

    /**
     * Actual distance query on the Data
     */
    private DistanceQuery<O> distQuery;

    /**
     * Random factory for selecting vantage points
     */
    RandomFactory random;

    /**
     * Sample size for selecting vantage points
     */
    int sampleSize;

    /**
     * Truncation parameter
     */
    int truncate;

    /**
     * Counter for distance computations.
     */
    long distComputations = 0L;

    /**
     * Root node from the tree
     */
    Node root;

    // TODO: not less than two!
    /**
     * The k Value to split to
     */
    static int kVal;

    // TODO: Implement standart vals somewhere?
  /**
   * Constructor with default values, used by EmpiricalQueryOptimizer
   * 
   * @param relation data for tree construction
   * @param distance distance function for tree construction
   * @param leafsize Leaf size and sample size (simpler parameterization)
   * @param kVal 
   */
  public VPkTree(Relation<O> relation, Distance<? super O> distance, int leafsize, int kVal) {
    this(relation, distance, RandomFactory.DEFAULT, leafsize, leafsize, kVal);
  }

    /**
     * Constructor.
     *
     * @param relation data for tree construction
     * @param distance distance function for tree construction
     * @param random Random generator for sampling
     * @param sampleSize Sample size for finding the vantage point
     * @param truncate Leaf size threshold
     * @param kVal k-split Value
     */
    public VPkTree(Relation<O> relation, Distance<? super O> distance, RandomFactory random, int sampleSize, int truncate, int kVal) {
        this.relation = relation;
        this.distFunc = distance;
        this.random = random;
        this.distQuery = distance.instantiate(relation);
        this.sampleSize = Math.max(sampleSize, 1);
        this.truncate = Math.max(truncate, 1);
        this.kVal = kVal;
    }

    @Override
    public void initialize() {
        root = new Builder().buildTree(0, relation.size());
    }

    /**
     * Build the VPk-Tree
     * 
     * @author Sebastian Aloisi
     */
    private class Builder {
        /**
         * Scratch space for organizing the elements
         */
        ModifiableDoubleDBIDList scratch;

        /**
         * Scratch iterator
         */
        DoubleDBIDListMIter scratchit;

        /**
         * Random generator
         */
        Random rnd;

        /**
         * Constructor.
         */
        public Builder() {
            scratch = DBIDUtil.newDistanceDBIDList(relation.size());
            for(DBIDIter it = relation.iterDBIDs(); it.valid(); it.advance()) {
                scratch.add(Double.NaN, it);
            }
            scratchit = scratch.iter();
            rnd = VPkTree.this.random.getSingleThreadedRandom();
        }

        /**
         * Build the tree recursively
         * 
         * @param left Left bound in scratch
         * @param right Right bound in scratch
         * @return new node
         */
        private Node buildTree(int left, int right) {
            assert left < right;
            if(left + truncate >= right) {
                DBID vp = DBIDUtil.deref(scratchit.seek(left));
                ModifiableDoubleDBIDList vps = DBIDUtil.newDistanceDBIDList(right - left);
                vps.add(0., vp);
                for(scratchit.advance(); scratchit.getOffset() < right; scratchit.advance()) {
                    vps.add(distance(vp, scratchit), scratchit);
                }
                return new Node(vps);
            }
            DBIDVar vantagePoint = chooseVantagePoint(left, right);
            int tied = 0;
            // Compute all the distances to the best vantage point (not just
            // sample)
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                if(DBIDUtil.equal(scratchit, vantagePoint)) {
                    scratchit.setDouble(0);
                    if(tied > 0 && scratchit.getOffset() != left + tied) {
                        scratch.swap(left, left + tied);
                    }
                    scratch.swap(scratchit.getOffset(), left);
                    tied++;
                    continue;
                }
                final double d = distance(scratchit, vantagePoint);
                scratchit.setDouble(d);
                if(d == 0) {
                    scratch.swap(scratchit.getOffset(), left + tied++);
                }
            }
            assert tied > 0;
            assert DBIDUtil.equal(vantagePoint, scratchit.seek(left)) : "tied: " + tied;
            // Note: many duplicates of vantage point:
            if(left + tied + truncate > right) {
                ModifiableDoubleDBIDList vps = DBIDUtil.newDistanceDBIDList(right - left);
                for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                    vps.add(scratchit.doubleValue(), scratchit);
                }
                return new Node(vps);
            }

            int quantilesAmount = kVal - 1;
            int[] quantiles = new int[quantilesAmount];

            int length = left + tied + right;

            double leftLowBound = Double.POSITIVE_INFINITY;
            double rightLowBound = Double.POSITIVE_INFINITY;
            double leftHighBound = Double.NEGATIVE_INFINITY;
            double rightHighBound = Double.NEGATIVE_INFINITY;

            for(int i = 0; i < quantilesAmount; i++) {
                double currentBound = (double) (i + 1) / (double) kVal;
                quantiles[i] = (int) ((double) length * currentBound);

                // TODO: falsch?
                QuickSelectDBIDs.quickSelect(scratch, left + tied, right, quantiles[i]);
                final double quantileDistVal = scratch.doubleValue(quantiles[i]);

                // offset for values == quantileDistVal, such that correct
                // sorting is given
                for(scratchit.seek(left + tied); scratchit.getOffset() < quantiles[i]; scratchit.advance()) {
                    final double d = scratchit.doubleValue();
                    // Move all tied with the quantile to the next partition
                    if(d == quantileDistVal) {
                        scratch.swap(scratchit.getOffset(), --quantiles[i]);
                        continue;
                    }
                    leftLowBound = d < leftLowBound ? d : leftLowBound;
                    leftHighBound = d > leftHighBound ? d : leftHighBound;
                }

                for(scratchit.seek(quantiles[i]); scratchit.getOffset() < right; scratchit.advance()) {
                    final double d = scratchit.doubleValue();
                    rightLowBound = d < rightLowBound ? d : rightLowBound;
                    rightHighBound = d > rightHighBound ? d : rightHighBound;
                }

                assert right > quantiles[i];
            }

            // Recursive build, include ties with parent:
            ModifiableDoubleDBIDList vps = DBIDUtil.newDistanceDBIDList(tied);
            for(scratchit.seek(left); scratchit.getOffset() < left + tied; scratchit.advance()) {
                vps.add(scratchit.doubleValue(), scratchit);
            }

            Node current = new Node(vps);

            // Recursive build the first kVal-1 child Partititions
            for(int i = 0; i < quantilesAmount; i++) {
                if(left + tied < quantiles[i]) {
                    current.children[i] = buildTree(left + tied, quantiles[i]);
                    current.children[i].lowBound = leftLowBound;
                    current.children[i].highBound = leftHighBound;
                }
            }

            // Build Child for the last partitition
            current.children[kVal] = buildTree(quantiles[quantilesAmount - 1], right);
            current.children[kVal].lowBound = rightLowBound;
            current.children[kVal].highBound = rightHighBound;
            return current;
        }

        // TODO: interface for VP Selection alternatives
        /**
         * Find a vantage points in the DBIDs between left and right
         * 
         * @param left Left bound in scratch
         * @param right Right bound in scratch
         * @return vantage point
         */
        private DBIDVar chooseVantagePoint(int left, int right) {
            // Random sampling:
            if(sampleSize == 1) {
                return scratch.assignVar(left + rnd.nextInt(right - left), DBIDUtil.newVar());
            }
            final int s = Math.min(sampleSize, right - left);
            double bestSpread = Double.NEGATIVE_INFINITY;
            DBIDVar best = DBIDUtil.newVar();
            // Modifiable copy for sampling:
            ArrayModifiableDBIDs workset = DBIDUtil.newArray(right - left);
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                workset.add(scratchit);
            }
            for(DBIDMIter it = DBIDUtil.randomSample(workset, s, rnd).iter(); it.valid(); it.advance()) {
                // Sample s+1 objects in case `it` is contained.
                DBIDUtil.randomShuffle(workset, rnd, Math.min(s + 1, workset.size()));
                double spread = calcMoment(it, workset, s);
                if(spread > bestSpread) {
                    bestSpread = spread;
                    best.set(it);
                }
            }
            return best;
        }

        // TODO: interface for VP Selection alternatives
        /**
         * Calculate the 2nd moment to the median of the distances to p
         * 
         * @param p DBID to calculate the moment for
         * @param check points to check with
         * @param size Maximum size to use
         * @return second moment
         */
        private double calcMoment(DBIDRef p, DBIDs check, int size) {
            double[] dists = new double[Math.min(size, check.size())];
            int i = 0;
            for(DBIDIter iter = check.iter(); iter.valid() && i < size; iter.advance()) {
                if(!DBIDUtil.equal(iter, p)) {
                    dists[i++] = distance(p, iter);
                }
            }
            double median = QuickSelect.median(dists);
            double ssq = 0;
            for(int j = 0; j < i; j++) {
                final double o = dists[j] - median;
                ssq += o * o;
            }
            return ssq / i;
        }

    }

    /**
     * The Node Class saves the important information for the each Node
     * 
     * @author Sebastian Aloisi
     * 
     *         Based on VPTree.Node written by Robert Gehde and Erich Schubert
     */
    protected static class Node {
        /**
         * Vantage point and singletons
         */
        ModifiableDoubleDBIDList vp;

        /**
         * child trees
         */
        Node[] children;

        /**
         * upper and lower distance bounds
         */
        double lowBound, highBound;

        /**
         * Constructor.
         * 
         * @param vp Vantage point and singletons
         */
        public Node(ModifiableDoubleDBIDList vp) {
            this.vp = vp;
            this.children = new Node[kVal];
            this.lowBound = Double.NaN;
            this.highBound = Double.NaN;

            assert !vp.isEmpty();
        }
    }

    /**
     * Compute a distance, and count.
     *
     * @param a First object
     * @param b Second object
     * @return Distance
     */
    private double distance(DBIDRef a, DBIDRef b) {
        ++distComputations;
        return distQuery.distance(a, b);
    }

    /**
     * Compute a distance, and count.
     *
     * @param a First object
     * @param b Second object
     * @return Distance
     */
    private double distance(O a, DBIDRef b) {
        ++distComputations;
        return distQuery.distance(a, b);
    }

    // TODO: Override interfaces

    @Override
    public PrioritySearcher<O> priorityByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'priorityByObject'");
    }

    // TODO: Implement Queries

}
