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
import elki.index.Index;
import elki.index.IndexFactory;
import elki.index.tree.metrical.vptree.VPTree.VPTreeKNNDBIDSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreeKNNObjectSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreePriorityDBIDSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreePriorityObjectSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreeRangeDBIDSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreeRangeObjectSearcher;
import elki.logging.Logging;
import elki.logging.statistics.LongStatistic;
import elki.math.MeanVariance;
import elki.utilities.Alias;
import elki.utilities.datastructures.heap.DoubleObjectMinHeap;
import elki.utilities.documentation.Reference;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.CommonConstraints;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.DoubleParameter;
import elki.utilities.optionhandling.parameters.EnumParameter;
import elki.utilities.optionhandling.parameters.IntParameter;
import elki.utilities.optionhandling.parameters.ObjectParameter;
import elki.utilities.optionhandling.parameters.RandomParameter;
import elki.utilities.random.RandomFactory;

/**
 * Generalized hyperplane Tree
 * <p>
 * Uses two Vantae Points for Generalized Hyperplane decomposition to split the
 * relation.
 * <p>
 * Reference:
 * <p>
 * J. K. Uhlmann <br>
 * Satisfying general proximity/similarity queries with metric trees<br>
 * Information Processing Letters 40(1991) 175-179
 * 
 * @author Sebastian Aloisi
 * @since 0.8.1
 * @param <O> Object Type
 */
@Reference(authors = "J. K. Uhlmann", //
        title = "Satisfying general proximity/similarity queries with metric trees", //
        booktitle = "Information Processing Letters 40(1991) 175-179", //
        url = "https://www.sciencedirect.com/science/article/abs/pii/002001909190074R", //
        bibkey = "DBLP:journals/ipl/Uhlmann91")

public class GHTree<O> implements DistancePriorityIndex<O> {
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
     * Maximumvariance threshold
     */

     double mvAlpha;

     /**
      * Vantage Point Selection Algorithm
      */
     VPSelectionAlgorithm vpSelector;

    /**
     * Counter for distance computations.
     */
    long distComputations = 0L;

    /**
     * Root node from the tree
     */
    Node root;


    /**
     * Constructor with default values, used by EmpiricalQueryOptimizer
     * 
     * @param relation data for tree construction
     * @param distance distance function for tree construction
     * @param leafsize Leaf size and sample size (simpler parameterization)
     * @param mvAlpha Maximum Variance threshold
     * @param vpSelector Vantage Point selection Algorithm
     */
    public GHTree(Relation<O> relation, Distance<? super O> distance, int leafsize, double mvAlpha, VPSelectionAlgorithm vpSelector) {
        this(relation, distance, RandomFactory.DEFAULT, leafsize, leafsize, mvAlpha, vpSelector);
    }

    /**
     * Constructor.
     *
     * @param relation data for tree construction
     * @param distance distance function for tree construction
     * @param random Random generator for sampling
     * @param sampleSize Sample size for finding the vantage point
     * @param truncate Leaf size threshold
     * @param mvAlpha Maximum Variance threshold
     * @param vpSelector Vantage Point selection Algorithm
     */
    public GHTree(Relation<O> relation, Distance<? super O> distance, RandomFactory random, int sampleSize, int truncate, double mvAlpha, VPSelectionAlgorithm vpSelector) {
        this.relation = relation;
        this.distFunc = distance;
        this.random = random;
        this.distQuery = distance.instantiate(relation);
        this.sampleSize = Math.max(sampleSize, 1);
        this.truncate = Math.max(truncate, 1);
        this.mvAlpha = mvAlpha;
        this.vpSelector = vpSelector;
    }

    @Override
    public void initialize() {
        root = new Builder().buildTree(0, relation.size());
    }

    private enum VPSelectionAlgorithm {
        RANDOM, FFT, MAXIMUM_VARIANCE, MAXIMUM_VARIANCE_SAMPLING, MAXIMUM_VARIANCE_FFT, REF_CHOOSE_VP
    }

    /**
     * check intersection of 2 intervals
     * 
     * @param l1 first lower bound
     * @param u1 first upper bound
     * @param l2 second lower bound
     * @param u2 second upper bound
     * @return if intervals intersect
     */
    private static boolean intersect(double l1, double u1, double l2, double u2) {
        return l1 <= u2 && l2 <= u1;
    }

    /**
     * Build the GH Tree
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
            rnd = GHTree.this.random.getSingleThreadedRandom();
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
                ModifiableDoubleDBIDList leftVps = DBIDUtil.newDistanceDBIDList(right - left);
                ModifiableDoubleDBIDList rightVps = DBIDUtil.newDistanceDBIDList();
                leftVps.add(0., vp);
                for(scratchit.advance(); scratchit.getOffset() < right; scratchit.advance()) {
                    leftVps.add(distance(vp, scratchit), scratchit);
                }
                return new Node(leftVps, rightVps);
            }

            DBIDVarTuple tuple;
            DBIDVar firstVP;
            DBIDVar secondVP;

            // TODO: check for all equal and/or less than 2
            // TODO: Ref selection?
            switch (GHTree.this.vpSelector) {
                case RANDOM:
                    tuple = selectRandomVantagePoints(left, right);
                    firstVP = tuple.first;
                    secondVP = tuple.second;
                    break;
                case FFT:
                    tuple = selectFFTVantagePoints(left, right);
                    firstVP = tuple.first;
                    secondVP = tuple.second;
                    break;
                case MAXIMUM_VARIANCE:
                    tuple = selectMaximumVarianceVantagePoints(left, right);
                    firstVP = tuple.first;
                    secondVP = tuple.second;
                    break;
                case MAXIMUM_VARIANCE_SAMPLING:
                    tuple = selectSampledMaximumVarianceVantagePoints(left, right);
                    firstVP = tuple.first;
                    secondVP = tuple.second;
                    break;
                case MAXIMUM_VARIANCE_FFT:
                    tuple = selectMVFFTVantagePoints(left, right);
                    firstVP = tuple.first;
                    secondVP = tuple.second;
                    break;
                default:
                    tuple = selectFFTVantagePoints(left, right);
                    firstVP = tuple.first;
                    secondVP = tuple.second;
                    break;
            }

            
            // TODO: what if secondVP = null

            //assert !DBIDUtil.equal(firstVP, secondVP);
            int tiedFirst = 0;
            int tiedSecond = 0;
            int firstPartititionSize = 0;

            // Compute difference between distances to Vantage Points
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                double firstDistance = distance(scratchit, firstVP);
                double secondDistance = distance(scratchit, secondVP);

                double distDiff = (firstDistance - secondDistance) / 2;

                if(DBIDUtil.equal(scratchit, firstVP)) {
                    scratchit.setDouble(firstDistance / 2);
                    if(tiedFirst > 0 && scratchit.getOffset() != left + tiedFirst) {
                        scratch.swap(left, left + tiedFirst);
                    }
                    scratch.swap(scratchit.getOffset(), left);
                    tiedFirst++;
                    continue;
                }

                if(DBIDUtil.equal(scratchit, secondVP)) {
                    scratchit.setDouble(secondDistance / 2);
                    if(tiedSecond > 0 && scratchit.getOffset() != right - tiedSecond) {
                        scratch.swap(right - 1, right - 1 - tiedSecond);
                    }
                    scratch.swap(scratchit.getOffset(), right - 1);
                    tiedSecond++;
                    continue;
                }

                if(distDiff < 0) {
                    firstPartititionSize += 1;
                }

                scratchit.setDouble(distDiff);

                if(distDiff == firstDistance / 2) {
                    scratch.swap(scratchit.getOffset(), left);
                }

                if(distDiff == secondDistance / 2) {
                    scratch.swap(scratchit.getOffset(), right - 1);
                }
            }

            assert tiedFirst > 0;
            assert tiedSecond > 0;
            assert DBIDUtil.equal(firstVP, scratchit.seek(left)) : "tiedFirst" + tiedFirst;
            if( secondVP != null){
                assert DBIDUtil.equal(secondVP, scratchit.seek(right - 1)) : "tiedSecond" + tiedSecond;
            }
            //assert (right - left) == firstPartititionSize + secondPartititionSize;

            // Note: many duplicates of vantage point:
            if(left + tiedFirst + truncate > right) {
                ModifiableDoubleDBIDList vps = DBIDUtil.newDistanceDBIDList(right - left);
                ModifiableDoubleDBIDList emptyVps = DBIDUtil.newDistanceDBIDList();
                for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                    vps.add(scratchit.doubleValue(), scratchit);
                }
                return new Node(vps, emptyVps);
            }

            int firstPartititionLimit = firstPartititionSize + tiedFirst;

            double firstLowBound = Double.MAX_VALUE;
            double firstHighBound = -1;
            double secondLowBound = Double.MAX_VALUE;
            double secondHighBound = -1;

            // sort left < 0; right >= 0
            QuickSelectDBIDs.quickSelect(scratch, left + tiedFirst, right - tiedSecond, firstPartititionLimit);

            for(scratchit.seek(left + tiedFirst); scratchit.getOffset() < firstPartititionLimit; scratchit.advance()) {
                final double d = scratchit.doubleValue();
                // Move all tied to hyperplane to the second Partitition
                if(d == 0) {
                    scratch.swap(scratchit.getOffset(), --firstPartititionLimit);
                    continue;
                }

                firstLowBound = d < firstLowBound ? d : firstLowBound;
                firstHighBound = d > firstHighBound ? d : firstHighBound;
            }

            for(scratchit.seek(firstPartititionLimit); scratchit.getOffset() < right - tiedSecond; scratchit.advance()) {
                final double d = scratchit.doubleValue();

                secondLowBound = d < secondLowBound ? d : secondLowBound;
                secondHighBound = d > secondHighBound ? d : secondHighBound;
            }

            assert right > firstPartititionLimit;
            // Recursive build, include ties with parent:

            ModifiableDoubleDBIDList firstVPs = DBIDUtil.newDistanceDBIDList(tiedFirst);
            ModifiableDoubleDBIDList secondVPs = DBIDUtil.newDistanceDBIDList(tiedSecond);

            for(scratchit.seek(left); scratchit.getOffset() < left + tiedFirst; scratchit.advance()) {
                firstVPs.add(scratchit.doubleValue(), scratchit);
            }

            for(scratchit.seek(right - tiedSecond); scratchit.getOffset() < right; scratchit.advance()) {
                secondVPs.add(scratchit.doubleValue(), scratchit);
            }

            Node current = new Node(firstVPs, secondVPs);

            // TODO: disappearing left branches?
            if(left + tiedFirst < firstPartititionLimit) {
                current.leftChild = buildTree(left + tiedFirst, firstPartititionLimit);
                current.leftChild.firstLowBound = firstLowBound;
                current.leftChild.firstHighBound = firstHighBound;
            }

            current.rightChild = buildTree(firstPartititionLimit, right - tiedSecond);
            current.rightChild.secondLowBound = secondLowBound;
            current.rightChild.secondHighBound = secondHighBound;

            return current;
        }

        private class DBIDVarTuple {
            DBIDVar first;

            DBIDVar second;

            public DBIDVarTuple(DBIDVar first, DBIDVar second) {
                this.first = first;
                this.second = second;
            }
        }

        /**
         * Finds two Vantage Points using Farthest first Traversal
         * 
         * @param left Left bound in scratch
         * @param right Right Bound in scratch
         * @return two Vantage points
         */
        private DBIDVarTuple selectFFTVantagePoints(int left, int right) {
            // First VP = random
            DBIDVar first = scratch.assignVar(left + rnd.nextInt(right - left), DBIDUtil.newVar());

            // Second VP = farthest away from first
            DBIDVar second = DBIDUtil.newVar();
            double maxDist = Double.NEGATIVE_INFINITY;

            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                double distance = distance(first, scratchit);
                if(distance > maxDist) {
                    second.set(scratchit);
                    maxDist = distance;
                }
            }

            return new DBIDVarTuple(first, second);
        }

        /**
         * Returns two random Vantage Points. Ignores duplicates.
         * 
         * @param left Left bound in scratch
         * @param right Right Bound in scratch
         * @return two Vantage points
         */
        private DBIDVarTuple selectRandomVantagePoints(int left, int right) {
            // Select First VP at random
            DBIDVar first = scratch.assignVar(left + rnd.nextInt(right - left), DBIDUtil.newVar());
            DBIDVar second = DBIDUtil.newVar();

            // Modifiable copy for selecting:
            ArrayModifiableDBIDs workset = DBIDUtil.newArray(right - left);
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                workset.add(scratchit);
            }

            // remove all Datapoints eqaul to first VP from workset
            for(DBIDMIter it = workset.iter(); it.valid(); it.advance()){
                if (DBIDUtil.equal(it, first)){
                    it.remove();
                }
            }

            second.set(DBIDUtil.randomSample(workset, random));

            DBIDVarTuple result = new DBIDVarTuple(first, second);

            return result;
        }

        /**
         * Returns the two Vantage Points with maximum Variance
         * 
         * @param left Left bound in scratch
         * @param right Right Bound in scratch
         * @return two Vantage points
         */
        private DBIDVarTuple selectMaximumVarianceVantagePoints(int left, int right){
            DBIDVar firstVP = DBIDUtil.newVar();
            DBIDVar secondVP = DBIDUtil.newVar();
            DBIDVar currentDbid = DBIDUtil.newVar();

            double bestStandartDeviation = Double.NEGATIVE_INFINITY;
            double bestMean = 0;
            double secondBestStandartDeviation = Double.NEGATIVE_INFINITY;
            double maxDist = -1;

            // Modifiable copy for selecting:
            ArrayModifiableDBIDs workset = DBIDUtil.newArray(right - left);
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                workset.add(scratchit);
            }

            // Select first VP
            for(DBIDMIter it = workset.iter(); it.valid(); it.advance()){
                currentDbid.set(it);

                MeanVariance currentVariance = new MeanVariance();

                for(DBIDMIter jt = workset.iter(); jt.valid(); jt.advance()){
                    double currentDistance = distance(currentDbid, jt);

                    currentVariance.put(currentDistance);

                    if(currentDistance > maxDist){
                        maxDist = currentDistance;
                    }
                }

                double currentStandartDeviance = currentVariance.getSampleStddev();

                if(currentStandartDeviance > bestStandartDeviation) {
                    firstVP.set(it);
                    bestStandartDeviation = currentStandartDeviance;
                    bestMean = currentVariance.getMean();
                }
            }

            // Remove all candidates from workingset exceeding threshold
            // Also remove all duplicates
            double omega = GHTree.this.mvAlpha * maxDist;

            for(DBIDMIter it = workset.iter(); it.valid(); it.advance()) {
                if (Math.abs(distance(firstVP,it) - bestMean) > omega){
                    it.remove();
                }

                if(DBIDUtil.equal(it, firstVP)){
                    it.remove();
                }
            }

            // Select second VP
            for(DBIDMIter it = workset.iter(); it.valid(); it.advance()){
                currentDbid.set(it);

                MeanVariance currentVariance = new MeanVariance();

                for(DBIDMIter jt = workset.iter(); jt.valid(); jt.advance()) {
                    double currentDistance = distance(currentDbid, jt);

                    currentVariance.put(currentDistance);
                }

                double currentStandartDeviance = currentVariance.getSampleStddev();

                if(currentStandartDeviance > secondBestStandartDeviation) {
                    secondVP.set(it);
                    secondBestStandartDeviation = currentStandartDeviance;
                }
            }

            return new DBIDVarTuple(firstVP, secondVP);
        }

        /**
         * Select Maximum Variance Vantage Points using a random sampled subset of relation
         * 
         * @param left
         * @param right
         * @return Vantage Points
         */
        private DBIDVarTuple selectSampledMaximumVarianceVantagePoints(int left, int right){
            if(GHTree.this.sampleSize == 2){
                return this.selectRandomVantagePoints(left, right);
            }

            // Create Workset to sample from
            final int s = Math.min(sampleSize, right - left);
            ArrayModifiableDBIDs scratchCopy = DBIDUtil.newArray(right - left);
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                scratchCopy.add(scratchit);
            }

            ModifiableDBIDs workset = DBIDUtil.randomSample(scratchCopy, s, rnd);

            DBIDVar firstVP = DBIDUtil.newVar();
            DBIDVar secondVP = DBIDUtil.newVar();
            DBIDVar currentDbid = DBIDUtil.newVar();

            double bestStandartDeviation = Double.NEGATIVE_INFINITY;
            double bestMean = 0;
            double secondBestStandartDeviation = Double.NEGATIVE_INFINITY;
            double maxDist = -1;

            // Select first VP
            for(DBIDMIter it = workset.iter(); it.valid(); it.advance()) {
                currentDbid.set(it);

                MeanVariance currentVariance = new MeanVariance();

                for(DBIDMIter jt = workset.iter(); jt.valid(); jt.advance()) {
                    double currentDistance = distance(currentDbid, jt);

                    currentVariance.put(currentDistance);

                    if(currentDistance > maxDist) {
                        maxDist = currentDistance;
                    }
                }

                double currentStandartDeviance = currentVariance.getSampleStddev();

                if(currentStandartDeviance > bestStandartDeviation) {
                    firstVP.set(it);
                    bestStandartDeviation = currentStandartDeviance;
                    bestMean = currentVariance.getMean();
                }
            }

            // Remove all candidates from workingset exceeding threshold
            // Also remove all duplicates
            double omega = GHTree.this.mvAlpha * maxDist;

            for(DBIDMIter it = workset.iter(); it.valid(); it.advance()) {
                if(Math.abs(distance(firstVP, it) - bestMean) > omega) {
                    it.remove();
                }

                if(DBIDUtil.equal(it, firstVP)) {
                    it.remove();
                }
            }

            // Select second VP
            for(DBIDMIter it = workset.iter(); it.valid(); it.advance()) {
                currentDbid.set(it);

                MeanVariance currentVariance = new MeanVariance();

                for(DBIDMIter jt = workset.iter(); jt.valid(); jt.advance()) {
                    double currentDistance = distance(currentDbid, jt);

                    currentVariance.put(currentDistance);
                }

                double currentStandartDeviance = currentVariance.getSampleStddev();

                if(currentStandartDeviance > secondBestStandartDeviation) {
                    secondVP.set(it);
                    secondBestStandartDeviation = currentStandartDeviance;
                }
            }

            return new DBIDVarTuple(firstVP, secondVP);
        }

        /**
         * First VP is selected by Maximum Variance
         * Second VP is selected by FFT
         * 
         * @param left
         * @param right
         * @return Vantage Points
         */
        private DBIDVarTuple selectMVFFTVantagePoints(int left, int right){
            // Create Workset to sample from
            final int s = Math.min(sampleSize, right - left);
            ArrayModifiableDBIDs scratchCopy = DBIDUtil.newArray(right - left);
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()) {
                scratchCopy.add(scratchit);
            }

            DBIDVar firstVP = DBIDUtil.newVar();
            DBIDVar secondVP = DBIDUtil.newVar();
            DBIDVar currentDbid = DBIDUtil.newVar();

            double bestStandartDeviation = Double.NEGATIVE_INFINITY;
            double maxdist = -1;

            // Select first VP
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
                    firstVP.set(scratchit);
                    bestStandartDeviation = currentStandartDeviance;
                }
            }

            // Select second VP
            for(scratchit.seek(left); scratchit.getOffset() < right; scratchit.advance()){
                double curdist = distance(firstVP, scratchit);

                if(curdist > maxdist){
                    maxdist = curdist;
                    secondVP.set(scratchit);
                }
            }

            return new DBIDVarTuple(firstVP, secondVP);
        }
    }

    /**
     * The Node Class saves the important information for the each Node
     * 
     * @author Sebastian Aloisi
     *         based on Node Class for VPTree writtten by Robert Gehde and Erich
     *         Schubert
     */
    protected static class Node {
        /**
         * "Left" Vantage point and Singletons
         */
        ModifiableDoubleDBIDList leftVp;

        /**
         * "Right" Vantage point and Singletons
         */
        ModifiableDoubleDBIDList rightVp;

        /**
         * child Trees
         */
        Node leftChild, rightChild;

        /**
         * upper and lower distance bounds
         */
        double firstLowBound, firstHighBound, secondLowBound, secondHighBound;

        /**
         * Constructor.
         * 
         * @param vp Vantage point and singletons
         */
        public Node(ModifiableDoubleDBIDList leftVp, ModifiableDoubleDBIDList rightVp) {
            this.leftVp = leftVp;
            this.rightVp = rightVp;
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
                        new GHTreeKNNObjectSearcher() : null;
    }

    @Override
    public KNNSearcher<DBIDRef> kNNByDBID(DistanceQuery<O> distanceQuery, int maxk, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHTreeKNNDBIDSearcher() : null;
    }

    @Override
    public RangeSearcher<O> rangeByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHTreeRangeObjectSearcher() : null;
    }

    @Override
    public RangeSearcher<DBIDRef> rangeByDBID(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHTreeRangeDBIDSearcher() : null;
    }

    @Override
    public PrioritySearcher<O> priorityByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHTreePriorityObjectSearcher() : null;
    }

    @Override
    public PrioritySearcher<DBIDRef> priorityByDBID(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHTreePriorityDBIDSearcher() : null;
    }

    /**
     * kNN search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     */
    public static abstract class GHTreeKNNSearcher {
        /**
         * Recursive search function
         * 
         * @param knns Current kNN results
         * @param node Current node
         * @return New tau
         */
        protected double ghKNNSearch(KNNHeap knns, Node node) {
            DoubleDBIDListIter firstVP = node.leftVp.iter();
            DoubleDBIDListIter secondVP = node.rightVp.iter();

            final double firstVPDistance = queryDistance(firstVP);
            // TODO: if not null
            final double secondVPDistance = queryDistance(secondVP);

            knns.insert(firstVPDistance, firstVP);
            knns.insert(secondVPDistance, secondVP);

            for(firstVP.advance(); firstVP.valid(); firstVP.advance()) {
                knns.insert(queryDistance(firstVP), firstVP);
            }

            for(secondVP.advance(); secondVP.valid(); secondVP.advance()) {
                knns.insert(queryDistance(secondVP), secondVP);
            }

            Node lc = node.leftChild, rc = node.rightChild;
            double tau = knns.getKNNDistance();

            final double firstDistanceDiff = (firstVPDistance - secondVPDistance) / 2;
            final double secondDistanceDiff = (secondVPDistance - firstDistanceDiff) / 2;

            // TODO: Priortization?

            if(lc != null && firstDistanceDiff < tau) {
                tau = ghKNNSearch(knns, lc);
            }

            if(rc != null && secondDistanceDiff < tau) {
                tau = ghKNNSearch(knns, rc);
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

    public class GHTreeKNNObjectSearcher extends GHTreeKNNSearcher implements KNNSearcher<O> {
        private O query;

        @Override
        public KNNList getKNN(O query, int k) {
            final KNNHeap knns = DBIDUtil.newHeap(k);
            this.query = query;
            ghKNNSearch(knns, root);
            return knns.toKNNList();
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHTree.this.distance(query, p);
        }
    }

    /**
     * 
     * kNN search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     */
    public class GHTreeKNNDBIDSearcher extends GHTreeKNNSearcher implements KNNSearcher<DBIDRef> {

        private DBIDRef query;

        @Override
        public KNNList getKNN(DBIDRef query, int k) {
            final KNNHeap knns = DBIDUtil.newHeap(k);
            this.query = query;
            ghKNNSearch(knns, root);
            return knns.toKNNList();
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHTree.this.distance(query, p);
        }
    }

    /**
     * Range Searcher for the GH-Tree
     */
    public static abstract class GHTreeRangeSearcher {

        protected void ghRangeSearch(ModifiableDoubleDBIDList result, Node node, double range) {
            final DoubleDBIDListIter firstVP = node.leftVp.iter();
            final DoubleDBIDListMIter secondVP = node.rightVp.iter();

            final double firstVPDistance = queryDistance(firstVP);
            // TODO: if not null
            final double secondVPDistance = queryDistance(secondVP);

            if(firstVPDistance < range) {
                result.add(firstVPDistance, firstVP);
            }

            for(firstVP.advance(); firstVP.valid(); firstVP.advance()) {
                final double d = queryDistance(firstVP);
                if(d <= range) {
                    result.add(d, firstVP);
                }
            }

            // TODO: if both, smarter ?

            if(secondVPDistance < range) {
                result.add(secondVPDistance, secondVP);
            }

            for(secondVP.advance(); secondVP.valid(); secondVP.advance()) {
                final double d = queryDistance(secondVP);
                if(d <= range) {
                    result.add(d, secondVP);
                }
            }

            Node lc = node.leftChild, rc = node.rightChild;

            final double firstDistanceDiff = (firstVPDistance - secondVPDistance) / 2;
            final double secondDistanceDiff = (secondVPDistance - firstDistanceDiff) / 2;

            if(lc != null && firstDistanceDiff < range) {
                ghRangeSearch(result, lc, range);
            }

            if(rc != null && secondDistanceDiff < range) {
                ghRangeSearch(result, rc, range);
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
     * Range search for the GH-Tree.
     * 
     * @author Sebastian Aloisi
     */
    public class GHTreeRangeObjectSearcher extends GHTreeRangeSearcher implements RangeSearcher<O> {

        private O query;

        @Override
        public ModifiableDoubleDBIDList getRange(O query, double range, ModifiableDoubleDBIDList result) {
            this.query = query;
            ghRangeSearch(result, root, range);
            return result;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHTree.this.distance(query, p);
        }
    }

    /**
     * Range search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     */
    public class GHTreeRangeDBIDSearcher extends GHTreeRangeSearcher implements RangeSearcher<DBIDRef> {

        private DBIDRef query;

        @Override
        public ModifiableDoubleDBIDList getRange(DBIDRef query, double range, ModifiableDoubleDBIDList result) {
            this.query = query;
            ghRangeSearch(result, root, range);
            return result;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHTree.this.distance(query, p);
        }
    }

    /**
     * Priority search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     * 
     * @param <Q> query type
     */
    public abstract class GHTreePrioritySearcher<Q> implements PrioritySearcher<Q> {
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
        private double curdist, vpDist;

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
            if(candidates.valid()){
                candidates.advance();
            }

            do {
                while(candidates.valid()){
                    if(vpDist - candidates.doubleValue() <= threshold){
                        return this;
                    }
                    candidates.advance();
                }
            }while(advanceQueue());
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
            heap.poll();

            DoubleDBIDListIter firstCandidates = cur.leftVp.iter();
            double firstDist = queryDistance(firstCandidates);
            Node lc = cur.leftChild;

            if(lc != null && intersect(firstDist - threshold, firstDist + threshold, lc.firstLowBound, lc.firstHighBound)) {
                final double mindist = Math.max(Math.max(firstDist - lc.firstHighBound, lc.firstLowBound - firstDist), curdist);
                if(mindist <= threshold) {
                    heap.add(mindist, lc);
                }
            }

            if (cur.rightVp == null || cur.rightVp.isEmpty()){
                candidates = firstCandidates;
                vpDist = firstDist;
                return true;
            }

            DoubleDBIDListIter secondCandidates = cur.rightVp.iter();
            double secondDist = queryDistance(secondCandidates);

            Node rc = cur.rightChild;

            if(rc != null && intersect(secondDist - threshold, secondDist + threshold, rc.firstLowBound, rc.firstHighBound)) {
                final double mindist = Math.max(Math.max(secondDist - rc.firstHighBound, rc.firstLowBound - secondDist), curdist);
                if(mindist <= threshold) {
                    heap.add(mindist, rc);
                }
            }

            if (firstDist < secondDist){
                candidates = firstCandidates;
                vpDist = firstDist;
                return true;
            }

            candidates = secondCandidates;
            vpDist = secondDist;
            return true;
        }

        /**
         * Query the distance to a query object.
         *
         * @param iter Target object
         * @return Distance
         */
        protected abstract double queryDistance(DBIDRef iter);

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
            return candidates.doubleValue() == 0. ? vpDist : queryDistance(candidates);
        }

        @Override
        public double getApproximateDistance() {
            return vpDist;
        }

        @Override
        public double getApproximateAccuracy() {
            return candidates.doubleValue();
        }

        @Override
        public double getLowerBound() {
            return Math.max(vpDist - candidates.doubleValue(), curdist);
        }

        @Override
        public double getUpperBound() {
            return vpDist + candidates.doubleValue();
        }

        @Override
        public double allLowerBound() {
            return curdist;
        }
    }

    /**
     * Priority Range Search for the GH-tree
     * 
     * @author Sebastian Aloisi
     * 
     */
    public class GHTreePriorityObjectSearcher extends GHTreePrioritySearcher<O> {
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
            return GHTree.this.distance(query, p);
        }
    }

    /**
     * Priority Range Search for the GH-tree
     * 
     * @author Sebastian Aloisi
     * 
     */
    public class GHTreePriorityDBIDSearcher extends GHTreePrioritySearcher<DBIDRef>{
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
            return GHTree.this.distance(query, p);
        }
    }

    @Override
    public void logStatistics() {
        LOG.statistics(new LongStatistic(this.getClass().getName() + ".distance-computations", distComputations));
    }

    /**
     * Index factory for the GH-Tree
     * 
     * @author Sebastian Aloisi
     * 
     * @param <O> Object type
     */
    @Alias({ "gh" })
    public static class Factory<O extends NumberVector> implements IndexFactory<O> {

        /**
         * Distance Function
         */
        Distance<? super O> distance;

        /**
         * Random Factory
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
         * Maximum Variance threshold
         */

         double mvAlpha;

        /**
         * Vantage Point selection Algorithm
         */
        VPSelectionAlgorithm vpSelector;

        /**
         * Constructor
         * 
         * @param distance distance function
         * @param random random generator
         * @param sampleSize sample size
         * @param truncate maximum leaf size (truncation)
         * @param mvAlpha Maximum Variance threshold
         * @param vpSelector Vantage Point selection Algorithm
         */
        public Factory(Distance<? super O> distance, RandomFactory random, int sampleSize, int truncate, double mvAlpha, VPSelectionAlgorithm vpSelector) {
            super();
            this.distance = distance;
            this.random = random;
            this.sampleSize = sampleSize;
            this.truncate = truncate;
            this.mvAlpha = mvAlpha;
            this.vpSelector = vpSelector;
        }

        @Override
        public Index instantiate(Relation<O> relation) {
            return new GHTree<>(relation, distance, random, truncate, sampleSize, mvAlpha, vpSelector);
        }

        @Override
        public TypeInformation getInputTypeRestriction() {
            return distance.getInputTypeRestriction();
        }

        /**
         * Parameterization class
         * 
         * @author Sebastian Aloisi
         */
        public static class Par<O extends NumberVector> implements Parameterizer {

            /**
             * Parameter to specify the distance function to determine the
             * distance
             * between database objects, must extend
             * {@link elki.distance.Distance}.
             */
            public final static OptionID DISTANCE_FUNCTION_ID = new OptionID("ghtree.distanceFunction", "Distance function to determine the distance between objects");

            /**
             * Parameter to specify the sample size for choosing vantage points
             */
            public final static OptionID SAMPLE_SIZE_ID = new OptionID("ghtree.sampleSize", "Size of sample to select vantage points from");

            /**
             * Parameter to specify the minimum leaf size
             */
            public final static OptionID TRUNCATE_ID = new OptionID("ghtree.truncate", "Minimum leaf size for stopping");
            
            /**
             * Parameter to specify Maximum Variance Threshold
             */
            public final static OptionID MV_ALPHA_ID = new OptionID("ghtree.mvAlpha","Threshold for Maximum Variance VP selection Algorithm");
            /**
             * Parameter to specify the rnd generator seed
             */
            public final static OptionID SEED_ID = new OptionID("ghtree.seed", "The rnd number generator seed");

            /**
             * Parameter to specify the Vantage Point selection Algorithm
             */
            public final static OptionID VPSELECTOR_ID = new OptionID("ghtree.vpSelector", "The Vantage Point selection Algorithm");

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
             * Maximum Variance Threshold Parameter
             */
            double mvAlpha;

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
                                LOG.warning("GHtree requires a metric to be exact.");
                            }
                        });
                new IntParameter(SAMPLE_SIZE_ID, 10) //
                        .addConstraint(CommonConstraints.GREATER_THAN_ONE_INT) //
                        .grab(config, x -> this.sampleSize = x);
                new IntParameter(TRUNCATE_ID, 8) //
                        .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
                        .grab(config, x -> this.truncate = x);
                new RandomParameter(SEED_ID).grab(config, x -> random = x);
                new DoubleParameter(MV_ALPHA_ID,0.5) //
                        .addConstraint(CommonConstraints.LESS_EQUAL_ONE_DOUBLE)
                        .addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE)
                        .grab(config, x -> this.mvAlpha = x);
                new EnumParameter<>(VPSELECTOR_ID, VPSelectionAlgorithm.class).grab(config, x -> this.vpSelector = x);
            }

            @Override
            public Object make() {
                return new Factory<>(distance, random, sampleSize, truncate,mvAlpha, vpSelector);
            }
        }
    }
}
