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

import elki.data.NumberVector;
import elki.data.type.TypeInformation;
import elki.database.ids.ArrayModifiableDBIDs;
import elki.database.ids.DBID;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDMIter;
import elki.database.ids.DBIDRef;
import elki.database.ids.DBIDUtil;
import elki.database.ids.DBIDVar;
import elki.database.ids.DBIDs;
import elki.database.ids.DoubleDBIDListIter;
import elki.database.ids.DoubleDBIDListMIter;
import elki.database.ids.KNNHeap;
import elki.database.ids.KNNList;
import elki.database.ids.ModifiableDBIDs;
import elki.database.ids.ModifiableDoubleDBIDList;
import elki.database.ids.QuickSelectDBIDs;
import elki.database.query.PrioritySearcher;
import elki.database.query.QueryBuilder;
import elki.database.query.distance.DistanceQuery;
import elki.database.query.knn.KNNSearcher;
import elki.database.query.range.RangeSearcher;
import elki.database.relation.Relation;
import elki.distance.Distance;
import elki.index.DistancePriorityIndex;
import elki.index.IndexFactory;
import elki.logging.Logging;
import elki.logging.statistics.LongStatistic;
import elki.math.MeanVariance;
import elki.utilities.Alias;
import elki.utilities.datastructures.QuickSelect;
import elki.utilities.datastructures.heap.DoubleObjectMinHeap;
import elki.utilities.documentation.Reference;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.EnumParameter;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import elki.utilities.optionhandling.parameters.RandomParameter;
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
    private static final Logging LOG = Logging.getLogger(VPkTree.class);

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
     * The k Value to split to
     */
    static int kVal;

    /**
     * Counter for distance computations.
     */
    long distComputations = 0L;

    /**
     * Root node from the tree
     */
    Node root;
    
    /**
     * Vantage Point Selection Algorithm
     */
    VPSelectionAlgorithm vpSelector;

    /**
     * Constructor with default values, used by EmpiricalQueryOptimizer
     * 
     * @param relation data for tree construction
     * @param distance distance function for tree construction
     * @param leafsize Leaf size and sample size (simpler parameterization)
     * @param kVal k-split Value
     * @param vpSelector Vantage Point selection Algorithm
     */
    public VPkTree(Relation<O> relation, Distance<? super O> distance, int leafsize, int kVal, VPSelectionAlgorithm vpSelector) {
        this(relation, distance, RandomFactory.DEFAULT, leafsize, leafsize, kVal, vpSelector);
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
     * @param vpSelector Vantage Point selection Algorithm
     */
    public VPkTree(Relation<O> relation, Distance<? super O> distance, RandomFactory random, int sampleSize, int truncate, int kVal, VPSelectionAlgorithm vpSelector) {
        this.relation = relation;
        this.distFunc = distance;
        this.random = random;
        this.distQuery = distance.instantiate(relation);
        this.sampleSize = Math.max(sampleSize, 1);
        this.truncate = Math.max(truncate, 1);
        this.kVal = kVal;
        this.vpSelector = vpSelector;
    }

    @Override
    public void initialize() {
        root = new Builder().buildTree(0, relation.size());
    }

    private enum VPSelectionAlgorithm {
        RANDOM,
        MAXIMUM_VARIANCE,
        MAXIMUM_VARIANCE_SAMPLING,
        REF_CHOOSE_VP
    }

    /**
     * Build the VPk-Tree
     * 
     * @author Sebastian Aloisi
     * based on Builder Class for VP-Tree written by Erich Schubert
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
            
            DBIDVar vantagePoint;

            switch (vpSelector){
                case MAXIMUM_VARIANCE: 
                    vantagePoint = selectMaximumVarianceVantagePoint(left, right);
                    break;
                case  MAXIMUM_VARIANCE_SAMPLING:
                    vantagePoint = selectSampledMaximumVarianceVantagePoint(left, right);
                    break;
                case REF_CHOOSE_VP:
                    vantagePoint = chooseVantagePoint(left, right);
                    break;
                default:
                    vantagePoint = selectRandomVantagePoint(left, right);
                    break;
            }

            int tied = 0;
            // Compute all the distances to the vantage point
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

            double lowBounds[] = new double[kVal];
            double highBounds[] = new double[kVal];

            for(int i = 0; i < kVal; i++) {
                lowBounds[i] = Double.POSITIVE_INFINITY;
                highBounds[i] = -1;
            }

            for(int i = 0; i < quantilesAmount; i++) {
                double currentBound = (double) (i + 1) / (double) kVal;
                quantiles[i] = QuickSelectDBIDs.quantile(scratch, left+tied, right, currentBound);
                
            }

            int leftQuant = left+tied;
            for(int i = 0; i < quantilesAmount; i++) {

                final double quantileDistVal = scratch.doubleValue(quantiles[i]);

                // offset for values == quantileDistVal, such that correct
                // sorting is given
                for(scratchit.seek(leftQuant); scratchit.getOffset() < quantiles[i]; scratchit.advance()) {
                    final double d = scratchit.doubleValue();
                    // Move all tied with the quantile to the next partition
                    if(d == quantileDistVal) {
                        scratch.swap(scratchit.getOffset(), --quantiles[i]);
                        continue;
                    }
                    lowBounds[i] = d < lowBounds[i] ? d : lowBounds[i];
                    highBounds[i] = d > highBounds[i] ? d : highBounds[i];
                }
                
                leftQuant = quantiles[i];

                assert right > quantiles[i];
            }

            for(scratchit.seek(quantiles[quantiles.length-1]); scratchit.getOffset() < right; scratchit.advance()) {
                final double d = scratchit.doubleValue();
                lowBounds[kVal - 1] = d < lowBounds[kVal - 1] ? d : lowBounds[kVal - 1];
                highBounds[kVal - 1] = d > highBounds[kVal - 1] ? d : highBounds[kVal - 1];
            }

            // Recursive build, include ties with parent:
            ModifiableDoubleDBIDList vps = DBIDUtil.newDistanceDBIDList(tied);
            for(scratchit.seek(left); scratchit.getOffset() < left + tied; scratchit.advance()) {
                vps.add(scratchit.doubleValue(), scratchit);
            }

            Node current = new Node(vps);
            
            int leftBound = left + tied;
            // Recursive build the first kVal-1 child Partititions
            for(int i = 0; i < kVal -1; i++) {
                    current.children[i] = buildTree(leftBound, quantiles[i]);
                    current.children[i].lowBound = lowBounds[i];
                    current.children[i].highBound = highBounds[i];
                    leftBound = quantiles[i];
            }

            // Build Child for the last partitition
            current.children[kVal - 1] = buildTree(quantiles[quantiles.length - 1], right);
            current.children[kVal - 1].lowBound = lowBounds[kVal - 1];
            current.children[kVal - 1].highBound = highBounds[kVal - 1];
            return current;
        }

        
        /**
         * Find a vantage points in the DBIDs between left and right
         * 
         * @param left Left bound in scratch
         * @param right Right bound in scratch
         * @return vantage point
         */
        private DBIDVar selectRandomVantagePoint(int left, int right) {
            scratchit.seek(left);
            DBIDVar result = DBIDUtil.newVar();
            ArrayModifiableDBIDs workset = DBIDUtil.newArray(right - left);
            
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                workset.add(scratchit);
            }

            DBIDMIter it = DBIDUtil.randomSample(workset, 1, rnd).iter();
            
            result.set(it);
            return result;
        }

        
        /**
         * Finds the Vantage Point with Maximum Variance to all other Data
         * Points
         * 
         * @param left
         * @param right
         * @return vantage point
         */
        private DBIDVar selectMaximumVarianceVantagePoint(int left, int right){

            DBIDVar best = DBIDUtil.newVar();
            DBIDVar currentDbid = DBIDUtil.newVar();
            double bestStandartDeviation = Double.NEGATIVE_INFINITY;

            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()){
                int currentOffset = scratchit.getOffset();
                
                currentDbid.set(scratchit);

                MeanVariance currentVariance = new MeanVariance();

                for (scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()){
                    double currentDistance = distance(currentDbid, scratchit);

                    currentVariance.put(currentDistance);
                }

                scratchit.seek(currentOffset);

                double currentStandartDeviance = currentVariance.getSampleStddev();

                if (currentStandartDeviance > bestStandartDeviation){
                    best.set(scratchit);
                    bestStandartDeviation = currentStandartDeviance;
                }
            }
            
            return best;
        }


        /**
         * Finds the Vantage Point with maximum Variance in respect to random
         * sampled subsets of relation.
         * 
         * @param left
         * @param right
         * @return vantage point
         */
        private DBIDVar selectSampledMaximumVarianceVantagePoint(int left, int right) {
            DBIDVar best = DBIDUtil.newVar();
            // Random sampling:
            if(sampleSize == 1) {
                return scratch.assignVar(left + rnd.nextInt(right - left), DBIDUtil.newVar());
            }

            final int s = Math.min(sampleSize, right - left);
            ArrayModifiableDBIDs workset = DBIDUtil.newArray(right - left);
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                workset.add(scratchit);
            }

            double bestStandartDeviation = Double.NEGATIVE_INFINITY;

            ModifiableDBIDs worksetDBIDs = DBIDUtil.randomSample(workset, s, rnd);

            for(DBIDMIter it = worksetDBIDs.iter(); it.valid(); it.advance()) {

                MeanVariance currentVariance = new MeanVariance();

                for(DBIDMIter jt = worksetDBIDs.iter(); jt.valid(); jt.advance()) {
                    double currentDistance = distance(it, jt);

                    currentVariance.put(currentDistance);
                }

                double currentStandartDeviance = currentVariance.getSampleStddev();

                if(currentStandartDeviance > bestStandartDeviation) {
                    best.set(it);
                    bestStandartDeviation = currentStandartDeviance;
                }
            }

            return best;

        }

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

    @Override
    public KNNSearcher<O> kNNByObject(DistanceQuery<O> distanceQuery, int maxk, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new VPkTreeKNNObjectSearcher() : null;
    }

    @Override
    public KNNSearcher<DBIDRef> kNNByDBID(DistanceQuery<O> distanceQuery, int maxk, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new VPkTreeKNNDBIDSearcher() : null;
    }

    @Override
    public RangeSearcher<O> rangeByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new VPkTreeRangeObjectSearcher() : null;
    }

    @Override
    public RangeSearcher<DBIDRef> rangeByDBID(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new VPkTreeRangeDBIDSearcher() : null;
    }

    @Override
    public PrioritySearcher<O> priorityByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new VPkTreePriorityObjectSearcher() : null;
    }

    @Override
    public PrioritySearcher<DBIDRef> priorityByDBID(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new VPkTreePriorityDBIDSearcher() : null;
    }

    /**
     * kNN search for the VPk-Tree
     * 
     * @author Sebastian Aloisi
     * 
     *         Based on VPTreeKNNSearcher for VP-Tree
     *         written by Robert Gehde and Erich Schubert
     */

    public static abstract class VPkTreeKNNSearcher {
        /**
         * Recursive search function
         * 
         * @param knns Current kNN results
         * @param node Current node
         * @return New tau
         */
        protected double vpkKNNSearch(KNNHeap knns, Node node) {
            DoubleDBIDListIter vp = node.vp.iter();
            final double vpDist = queryDistance(vp);
            knns.insert(vpDist, vp);
            for(vp.advance(); vp.valid(); vp.advance()) {
                knns.insert(queryDistance(vp), vp);
            }

            // Choose the Childnode to prioritize
            Node[] childNodes = node.children;
            double tau = knns.getKNNDistance();

            DoubleObjectMinHeap<Node> nodePrioHeap = new DoubleObjectMinHeap<>();

            for(int i = 0; i < childNodes.length ; i++){
                if (childNodes[i] != null){
                    Node currentNode = childNodes[i];
                    double prio = (currentNode.highBound -(vpDist + tau)) + (vpDist - tau  - currentNode.lowBound);
                    nodePrioHeap.add(prio, currentNode);
                }
            }

            while (!nodePrioHeap.isEmpty()){
                Node searchNode = nodePrioHeap.peekValue();

                if (searchNode.lowBound <= vpDist + tau && vpDist - tau <= searchNode.highBound){
                    tau = vpkKNNSearch(knns, searchNode);
                }
                nodePrioHeap.poll();
            }

            return tau;
        }

        /**
         * Compute the distance to a candidate object.
         * 
         * @param p Object
         * @return Distance
         */
        protected abstract double queryDistance(DBIDRef p);
    }

    /**
     * kNN search for the VPk-Tree.
     *
     * @author Sebastian Aloisi
     *         Based on VPTreeKNNObjectSearcher written by Robert Gehde and
     *         Erich Schubert
     */
    public class VPkTreeKNNObjectSearcher extends VPkTreeKNNSearcher implements KNNSearcher<O> {
        /**
         * Current query object
         */
        private O query;

        @Override
        public KNNList getKNN(O query, int k) {
            final KNNHeap knns = DBIDUtil.newHeap(k);
            this.query = query;
            vpkKNNSearch(knns, root);
            return knns.toKNNList();
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return VPkTree.this.distance(query, p);
        }
    }

    /**
     * kNN search for the VPk-Tree.
     *
     * @author Sebastian Aloisi
     *         Based on VPTreeKNNObjectSearcher written by Robert Gehde and
     *         Erich Schubert
     */
    public class VPkTreeKNNDBIDSearcher extends VPkTreeKNNSearcher implements KNNSearcher<DBIDRef> {
        /**
         * Current query object
         */
        private DBIDRef query;

        @Override
        public KNNList getKNN(DBIDRef query, int k) {
            final KNNHeap knns = DBIDUtil.newHeap(k);
            this.query = query;
            vpkKNNSearch(knns, root);
            return knns.toKNNList();
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return VPkTree.this.distance(query, p);
        }
    }

    /**
     * Range search for the VPk-Tree
     * 
     * @author Sebastian Aloisi
     *         based on VPTreeRangeSearcher written by Robert Gehde and Erich
     *         Schubert
     */
    public static abstract class VPkTreeRangeSearcher {
        /**
         * Recursive search function.
         *
         * @param result Result output
         * @param node Current node
         * @param range Search radius
         */
        protected void vpRangeSearch(ModifiableDoubleDBIDList result, Node node, double range) {
            final DoubleDBIDListMIter vp = node.vp.iter();
            final double x = queryDistance(vp);

            if(x <= range) {
                result.add(x, vp);
            }

            for(vp.advance(); vp.valid(); vp.advance()) {
                final double d = queryDistance(vp);
                if(d <= range) {
                    result.add(d, vp);
                }
            }

            Node[] children = node.children;

            for(int i = 0; i < kVal - 1; i++) {
                Node currentChild = children[i];

                if(currentChild != null && currentChild.lowBound <= x + range && x - range <= currentChild.highBound) {
                    vpRangeSearch(result, currentChild, range);
                }
            }

            Node currentChild = children[kVal - 1];
            if(currentChild != null && currentChild.lowBound <= x + range && x - range <= currentChild.highBound) {
                vpRangeSearch(result, currentChild, range);
            }
        }

        /**
         * Compute the distance to a candidate object.
         * 
         * @param p Object
         * @return Distance
         */
        protected abstract double queryDistance(DBIDRef p);
    }

    /**
     * Range search for the VPk-Tree
     * 
     * @author Sebastian Aloisi
     *         based on VPTreeRangeSearcher written by Robert Gehde and Erich
     *         Schubert
     */
    public class VPkTreeRangeObjectSearcher extends VPkTreeRangeSearcher implements RangeSearcher<O> {
        /**
         * Current query object
         */
        private O query;

        @Override
        public ModifiableDoubleDBIDList getRange(O query, double range, ModifiableDoubleDBIDList result) {
            this.query = query;
            vpRangeSearch(result, root, range);
            return result;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return VPkTree.this.distance(query, p);
        }
    }

    /**
     * Range search for the VP-tree.
     * 
     * @author Robert Gehde
     * @author Erich Schubert
     */
    public class VPkTreeRangeDBIDSearcher extends VPkTreeRangeSearcher implements RangeSearcher<DBIDRef> {
        /**
         * Current query object
         */
        private DBIDRef query;

        @Override
        public ModifiableDoubleDBIDList getRange(DBIDRef query, double range, ModifiableDoubleDBIDList result) {
            this.query = query;
            vpRangeSearch(result, root, range);
            return result;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return VPkTree.this.distance(query, p);
        }
    }

    /**
     * Priority search for the VPk-Tree.
     *  TODO: This method is not Tested nor debugged
     * @author Sebastian Aloisi
     *         Based on VPTreePrioritySearcher written by Robert Gehde and Erich
     *         Schubert
     *
     * @param <Q> query type
     */
    public abstract class VPkTreePrioritySearcher<Q> implements PrioritySearcher<Q> {
        /**
         * Min heap for searching.
         */
        private DoubleObjectMinHeap<Node> heap = new DoubleObjectMinHeap<>();

        /**
         * Stopping threshold.
         */
        private double threshold;

        /**
         * Current search position.
         */
        private Node cur;

        /**
         * Current iterator.
         */
        private DoubleDBIDListIter candidates = DoubleDBIDListIter.EMPTY;

        /**
         * Distance to the current object.
         */
        private double curdist, vpdist;

        /**
         * Start the search.
         */
        public void doSearch() {
            this.threshold = Double.POSITIVE_INFINITY;
            this.candidates = DoubleDBIDListIter.EMPTY;
            this.heap.clear();
            this.heap.add(0, root);
            advance();
        }

        @Override
        public PrioritySearcher<Q> advance() {
            // Advance the main iterator, if defined:
            if(candidates.valid()) {
                candidates.advance();
            }
            // Additional points stored in the node / leaf
            do {
                while(candidates.valid()) {
                    if(vpdist - candidates.doubleValue() <= threshold) {
                        return this;
                    }
                    candidates.advance();
                }
            }
            while(advanceQueue()); // Try next node
            return this;
        }

        /**
         * Expand the next node of the priority heap.
         * 
         * @return success
         */
        protected boolean advanceQueue() {
            if(heap.isEmpty()) {
                candidates = DoubleDBIDListIter.EMPTY;
                return false;
            }
            curdist = heap.peekKey();
            if(curdist > threshold) {
                heap.clear();
                candidates = DoubleDBIDListIter.EMPTY;
                return false;
            }
            cur = heap.peekValue();
            heap.poll(); // Remove
            candidates = cur.vp.iter();
            // Exact distance to vantage point:
            vpdist = queryDistance(candidates);

            Node[] childNodes = cur.children;

            // Add Child Nodes to the Heap

            for(int i = 0; i < childNodes.length; i++) {
                Node currentChild = childNodes[i];

                if(currentChild != null) {
                    final double mindist = Math.max(Math.max(vpdist - currentChild.highBound, currentChild.lowBound - vpdist), curdist);
                    if(mindist <= threshold) {
                        heap.add(mindist, currentChild);
                    }
                }
            }

            return true;
        }

        /**
         * Compute the distance to a candidate object.
         * 
         * @param p Object
         * @return Distance
         */
        protected abstract double queryDistance(DBIDRef p);

        @Override
        public int internalGetIndex() {
            return candidates.internalGetIndex();
        }

        @Override
        public boolean valid() {
            return candidates.valid();
        }

        @Override
        public PrioritySearcher<Q> decreaseCutoff(double threshold) {
            assert threshold <= this.threshold : "Thresholds must only decreasee.";
            this.threshold = threshold;
            if(threshold < curdist) { // No more results possible:
                heap.clear();
                candidates = DoubleDBIDListIter.EMPTY;
            }
            return this;
        }

        @Override
        public double computeExactDistance() {
            return candidates.doubleValue() == 0. ? vpdist : queryDistance(candidates);
        }

        @Override
        public double getApproximateDistance() {
            return vpdist;
        }

        @Override
        public double getApproximateAccuracy() {
            return candidates.doubleValue();
        }

        @Override
        public double getLowerBound() {
            return Math.max(vpdist - candidates.doubleValue(), curdist);
        }

        @Override
        public double getUpperBound() {
            return vpdist + candidates.doubleValue();
        }

        @Override
        public double allLowerBound() {
            return curdist;
        }
    }

    /**
     * Range search for for the VPk-Tree.
     * 
     * @author Sebastian Aloisi
     *         Based on VPTreePriorityObjectSearcher written by Robert Gehde and
     *         Erich Schubert
     *
     */
    public class VPkTreePriorityObjectSearcher extends VPkTreePrioritySearcher<O> {
        /**
         * Current query object
         */
        private O query;

        @Override
        public PrioritySearcher<O> search(O query) {
            this.query = query;
            doSearch();
            return this;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return VPkTree.this.distance(query, p);
        }
    }

    /**
     * Range search for for the VPk-Tree.
     * 
     * @author Sebastian Aloisi
     *         Based on VPTreePriorityDBIDSearcher written by Robert Gehde and
     *         Erich Schubert
     *
     */
    public class VPkTreePriorityDBIDSearcher extends VPkTreePrioritySearcher<DBIDRef> {
        /**
         * Current query object
         */
        private DBIDRef query;

        @Override
        public PrioritySearcher<DBIDRef> search(DBIDRef query) {
            this.query = query;
            doSearch();
            return this;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return VPkTree.this.distance(query, p);
        }
    }

    @Override
    public void logStatistics() {
        LOG.statistics(new LongStatistic(this.getClass().getName() + ".distance-computations", distComputations));
    }

    /**
     * Index factory for the VPk-Tree
     *
     * @author Sebastian Aloisi
     *         based on the Index Factory for the VP-Tree written by Robert
     *         Gehde
     *
     * @param <O> Object type
     */
    @Alias({ "vpk" })
    public static class Factory<O extends NumberVector> implements IndexFactory<O> {
        /**
         * Distance Function
         */
        Distance<? super O> distance;

        /**
         * Random factory
         */
        RandomFactory random;

        /**
         * Sample size
         */
        int sampleSize;

        /**
         * Truncation parameter
         */
        int truncate;

        /**
         * k-fold split parameter
         */
        int kFold;
        
        /**
         * Vantage Point selection Algorithm
         */
        VPSelectionAlgorithm vpSelector;

        /**
         * Constructor.
         * 
         * @param distFunc distance function
         * @param random random generator
         * @param sampleSize sample size
         * @param truncate maximum leaf size (truncation)
         * @param kFold split size
         * @param vpSelector Vantage Point selection Algorithm
         */
        public Factory(Distance<? super O> distFunc, RandomFactory random, int sampleSize, int truncate, int kFold, VPSelectionAlgorithm vpSelector) {
            super();
            this.distance = distFunc;
            this.random = random;
            this.sampleSize = Math.max(sampleSize, 1);
            this.truncate = Math.max(truncate, 1);
            this.kFold = kFold;
            this.vpSelector = vpSelector;
        }

        @Override
        public VPkTree<O> instantiate(Relation<O> relation) {
            return new VPkTree<>(relation, distance, random, sampleSize, truncate, kFold, vpSelector);
        }

        @Override
        public TypeInformation getInputTypeRestriction() {
            return distance.getInputTypeRestriction();
        }

        /**
         * Parameterization class.
         *
         * @author Sebastian Aloisi
         * 
         *         Based on Parameterization class for VP-Tree written by Robert
         *         Gehde
         */
        public static class Par<O extends NumberVector> implements Parameterizer {
            /**
             * Parameter to specify the distance function to determine the
             * distance
             * between database objects, must extend
             * {@link elki.distance.Distance}.
             */
            public final static OptionID DISTANCE_FUNCTION_ID = new OptionID("vpktree.distanceFunction", "Distance function to determine the distance between objects.");

            /**
             * Parameter to specify the sample size for choosing vantage point
             */
            public final static OptionID SAMPLE_SIZE_ID = new OptionID("vpktree.sampleSize", "Size of sample to select vantage point from.");

            /**
             * Parameter to specify the minimum leaf size
             */
            public final static OptionID TRUNCATE_ID = new OptionID("vpktree.truncate", "Minimum leaf size for stopping.");

            /**
             * Parameter to specify the rnd generator seed
             */
            public final static OptionID SEED_ID = new OptionID("vpktree.seed", "The rnd number generator seed.");

            /**
             * Parameter to specify the k-fold split size
             */
            public final static OptionID KFOLD_ID = new OptionID("vpktree.kfold", "The size to k-fold split to");

            /**
             * Parameter to specify the VP selection Algorithm
             */
            public final static OptionID VPSELECTOR_ID = new OptionID("vpktree.vpSelector", "The Vantage Point selection Algorithm");

            /**
             * Distance function
             */
            protected Distance<? super O> distance;

            /**
             * Random generator
             */
            protected RandomFactory random;

            /**
             * Sample size
             */
            protected int sampleSize;

            /**
             * Truncation parameter
             */
            int truncate;

            /**
             * k-fold split size parameter
             */
            int kFold;

            /**
             * Vantage Point selection Algorithm
             */
            VPSelectionAlgorithm vpSelector;

            @Override
            public void configure(Parameterization config) {
                new ObjectParameter<Distance<? super O>>(DISTANCE_FUNCTION_ID, Distance.class) //
                        .grab(config, x -> {
                            this.distance = x;
                            if(!distance.isMetric()) {
                                LOG.warning("VPkTree requires a metric to be exact.");
                            }
                        });
                new IntParameter(SAMPLE_SIZE_ID, 10) //
                        .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
                        .grab(config, x -> this.sampleSize = x);
                new IntParameter(TRUNCATE_ID, 8) //
                        .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
                        .grab(config, x -> this.truncate = x);
                new RandomParameter(SEED_ID).grab(config, x -> random = x);
                new IntParameter(KFOLD_ID, 2).addConstraint(CommonConstraints.GREATER_THAN_ONE_INT).grab(config, x -> this.kFold = x);
                new EnumParameter<>(VPSELECTOR_ID, VPSelectionAlgorithm.class).grab(config, x -> this.vpSelector = x);
            }

            @Override
            public Factory<O> make() {
                return new Factory<>(distance, random, sampleSize, truncate, kFold, vpSelector);
            }
        }
    }
}
