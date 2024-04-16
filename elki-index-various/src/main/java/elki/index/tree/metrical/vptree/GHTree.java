package elki.index.tree.metrical.vptree;

import java.util.Random;

import elki.data.NumberVector;
import elki.data.type.TypeInformation;
import elki.database.ids.ArrayModifiableDBIDs;
import elki.database.ids.DBID;
import elki.database.ids.DBIDArrayIter;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDMIter;
import elki.database.ids.DBIDRef;
import elki.database.ids.DBIDUtil;
import elki.database.ids.DBIDVar;
import elki.database.ids.DBIDs;
import elki.database.ids.KNNHeap;
import elki.database.ids.KNNList;
import elki.database.ids.ModifiableDBIDs;
import elki.database.ids.ModifiableDoubleDBIDList;
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
import elki.logging.Logging;
import elki.logging.statistics.LongStatistic;
import elki.math.MeanVariance;
import elki.utilities.Alias;
import elki.utilities.datastructures.heap.ComparableMinHeap;
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
     * Random Thread for selecting vantage points
     */
    Random randomThread;

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
     * TODO: delete truncate
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
        this.randomThread = random.getSingleThreadedRandom();
        this.distQuery = distance.instantiate(relation);
        this.sampleSize = Math.max(sampleSize, 1);
        this.truncate = Math.max(truncate, 1);
        this.mvAlpha = mvAlpha;
        this.vpSelector = vpSelector;
    }

    @Override
    public void initialize() {
        root = new Node();
        buildTree(root, relation.getDBIDs());
    }

    private enum VPSelectionAlgorithm {
        RANDOM, FFT, MAXIMUM_VARIANCE, MAXIMUM_VARIANCE_SAMPLING, MAXIMUM_VARIANCE_FFT, MAXIMUM_VARIANCE_FFT_SAMPLING, REF_CHOOSE_VP
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
     * Build the tree recursively
     * 
     * @param current current node to build
     * @param content data to index
     * @return new node
     */
    private void buildTree(Node current, DBIDs content) {

        DBIDVarTuple tuple;
        DBIDVar firstVP = DBIDUtil.newVar();
        DBIDVar secondVP = DBIDUtil.newVar();

        // TODO: check for all equal and/or less than 2
        // TODO: Ref selection?
        switch(GHTree.this.vpSelector){
        case RANDOM:
            tuple = selectRandomVantagePoints(content);
            firstVP.set(tuple.first);
            secondVP.set(tuple.second);
            break;
        case FFT:
            tuple = selectFFTVantagePoints(content);
            firstVP.set(tuple.first);
            secondVP.set(tuple.second);
            break;
        case MAXIMUM_VARIANCE:
            tuple = selectMaximumVarianceVantagePoints(content);
            firstVP.set(tuple.first);
            secondVP.set(tuple.second);
            break;
        case MAXIMUM_VARIANCE_SAMPLING:
            tuple = selectSampledMaximumVarianceVantagePoints(content);
            firstVP.set(tuple.first);
            secondVP.set(tuple.second);
            break;
        case MAXIMUM_VARIANCE_FFT:
            tuple = selectMVFFTVantagePoints(content);
            firstVP.set(tuple.first);
            secondVP.set(tuple.second);
            break;
        case MAXIMUM_VARIANCE_FFT_SAMPLING:
            tuple = selectSampledMVFFTVantagePoints(content);
            firstVP.set(tuple.first);
            secondVP.set(tuple.second);
            break;
        default:
            tuple = selectFFTVantagePoints(content);
            firstVP.set(tuple.first);
            secondVP.set(tuple.second);
            break;
        }

        current.firstVP = firstVP;

        // If second VP is empty, Leaf is reached, just set low/highbound
        // Else build childnodes
        if(secondVP.isEmpty()) {
            current.firstLowBound = 0;
            current.firstHighBound = 0;
        } else {
            current.secondVP = secondVP;

            double firstDistance;
            double secondDistance;

            ModifiableDBIDs[] children = new ModifiableDBIDs[2];

            for(DBIDIter iter = content.iter(); iter.valid(); iter.advance()) {
                // If the current position is a VP, this will be set to current
                // offset
                int vpOffset = -1;

                if(DBIDUtil.equal(firstVP, iter)) {
                    vpOffset = 0;
                }

                if(DBIDUtil.equal(secondVP, iter)) {
                    vpOffset = 1;
                }

                int childOffset = -1;

                firstDistance = distance(firstVP, iter);
                secondDistance = distance(secondVP, iter);
                if(firstDistance < secondDistance) {
                    childOffset = 0;
                }
                else {
                    childOffset = 1;
                }

                if(vpOffset == -1) {
                    if(children[childOffset] == null) {
                        children[childOffset] = DBIDUtil.newArray();
                    }
                    children[childOffset].add(iter);
                }
                current.firstLowBound = current.firstLowBound > firstDistance ? firstDistance : current.firstLowBound;
                current.firstHighBound = current.firstHighBound < firstDistance ? firstDistance : current.firstHighBound;
                current.secondLowBound = current.secondLowBound > secondDistance ? secondDistance : current.secondLowBound;
                current.secondHighBound = current.secondHighBound < secondDistance ? secondDistance : current.secondHighBound;
            }

            if(children[0] != null) {
                buildTree(current.firstChild = new Node(), children[0]);
            }

            if(children[1] != null) {
                buildTree(current.secondChild = new Node(), children[1]);
            }
        }
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
     * @param content Content to choose VP's from
     * @return two Vantage points
     */
    private DBIDVarTuple selectFFTVantagePoints(DBIDs content) {
        ArrayModifiableDBIDs contentArray = DBIDUtil.newArray(content);
        // First VP = random
        DBIDArrayIter contentIter = contentArray.iter();
        int pos = randomThread.nextInt(content.size());
        DBIDVar first = DBIDUtil.newVar();
        contentIter.seek(pos);
        first.set(contentIter);

        // Second VP = farthest away from first
        DBIDVar second = DBIDUtil.newVar();
        double maxDist = 0;

        for(contentIter.seek(0); contentIter.valid(); contentIter.advance()) {
            double distance = distance(first, contentIter);
            if(distance > maxDist) {
                second.set(contentIter);
                maxDist = distance;
            }
        }

        return new DBIDVarTuple(first, second);
    }

    /**
     * Returns two random Vantage Points. Ignores duplicates.
     * 
     * @param content Content to choose VP's from
     * @return two Vantage points
     */
    private DBIDVarTuple selectRandomVantagePoints(DBIDs content) {
        ArrayModifiableDBIDs workset = DBIDUtil.newArray(content);
        // First VP = random
        DBIDArrayIter worksetIter = workset.iter();
        int pos = randomThread.nextInt(content.size());
        DBIDVar first = DBIDUtil.newVar();
        worksetIter.seek(pos);
        first.set(worksetIter);

        // remove all Datapoints eqaul to first VP from workset
        for(DBIDMIter it = workset.iter(); it.valid(); it.advance()) {
            if(DBIDUtil.equal(it, first)) {
                it.remove();
            }
        }

        // Choose Second VP at Random from remaining DBID's
        DBIDVar second = DBIDUtil.newVar();
        if( workset.size() > 0){
            second.set(DBIDUtil.randomSample(workset, randomThread));
        }

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
    private DBIDVarTuple selectMaximumVarianceVantagePoints(DBIDs content) {
        DBIDVar firstVP = DBIDUtil.newVar();
        DBIDVar secondVP = DBIDUtil.newVar();
        DBIDVar currentDbid = DBIDUtil.newVar();

        // TODO: Truncate!
        if (content.size() == 1){
            DBIDIter it = content.iter();
            firstVP.set(it);
            return new DBIDVarTuple(firstVP, secondVP);
        }

        double bestStandartDeviation = Double.NEGATIVE_INFINITY;
        double bestMean = 0;
        double secondBestStandartDeviation = Double.NEGATIVE_INFINITY;
        double maxDist = -1;

        // Modifiable copy for selecting:
        ArrayModifiableDBIDs workset = DBIDUtil.newArray(content);

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
            if(Math.abs(distance(firstVP, it) - bestMean) > omega || DBIDUtil.equal(it, firstVP)) {
                it.remove();
            }
        }

        // if only one left, chose this as Second VP
        // else select according to algorithm
        // if none left, let Second VP be null
        if(workset.size() == 1) {
            DBIDMIter it = workset.iter();
            secondVP.set(it);
        }
        else {
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
        }
        return new DBIDVarTuple(firstVP, secondVP);
    }

    /**
     * Select Maximum Variance Vantage Points using a random sampled subset
     * of relation
     * 
     * @param content Content to choose VP's from
     * @return Vantage Points
     */
    private DBIDVarTuple selectSampledMaximumVarianceVantagePoints(DBIDs content) {
        if(GHTree.this.sampleSize == 2) {
            return this.selectRandomVantagePoints(content);
        }

        DBIDVar firstVP = DBIDUtil.newVar();
        DBIDVar secondVP = DBIDUtil.newVar();
        DBIDVar currentDbid = DBIDUtil.newVar();

        // TODO: Truncate!
        if(content.size() == 1) {
            DBIDIter it = content.iter();
            firstVP.set(it);
            return new DBIDVarTuple(firstVP, secondVP);
        }

        // Create Workset to sample from
        final int adjustedSampleSize = Math.min(sampleSize, content.size());
        ArrayModifiableDBIDs scratchCopy = DBIDUtil.newArray(content);

        ModifiableDBIDs workset = DBIDUtil.randomSample(scratchCopy, adjustedSampleSize, random);

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
            if(Math.abs(distance(firstVP, it) - bestMean) > omega || DBIDUtil.equal(it, firstVP)) {
                it.remove();
            }
        }

        // if only one left, chose this as Second VP
        // else select according to algorithm
        // if none left, let Second VP be null
        if(workset.size() == 1) {
            DBIDMIter it = workset.iter();
            secondVP.set(it);
        } else {
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
        }
        return new DBIDVarTuple(firstVP, secondVP);
    }

    /**
     * First VP is selected by Maximum Variance
     * Second VP is selected by FFT
     * 
     * @param content Content to choose VP's from
     * @return Vantage Points
     */
    private DBIDVarTuple selectMVFFTVantagePoints(DBIDs content) {
        // Create Workset to sample from
        ArrayModifiableDBIDs workset = DBIDUtil.newArray(content);

        DBIDVar firstVP = DBIDUtil.newVar();
        DBIDVar secondVP = DBIDUtil.newVar();
        DBIDVar currentDbid = DBIDUtil.newVar();

        if (content.size() == 1){
            DBIDIter iter = content.iter();
            firstVP.set(iter);
            return new DBIDVarTuple(firstVP, secondVP);
        }

        double bestStandartDeviation = Double.NEGATIVE_INFINITY;
        double maxdist = 0;

        // Select first VP
        for(DBIDArrayIter iter = workset.iter(); iter.valid(); iter.advance()) {
            int currentOffset = iter.getOffset();

            currentDbid.set(iter);

            MeanVariance currentVariance = new MeanVariance();

            for(iter.seek(0); iter.valid(); iter.advance()) {
                double currentDistance = distance(currentDbid, iter);

                currentVariance.put(currentDistance);
            }

            iter.seek(currentOffset);

            double currentStandartDeviance = currentVariance.getSampleStddev();

            if(currentStandartDeviance > bestStandartDeviation) {
                firstVP.set(iter);
                bestStandartDeviation = currentStandartDeviance;
            }
        }

        // Select second VP
        for(DBIDArrayIter iter = workset.iter(); iter.valid(); iter.advance()) {
            double curdist = distance(firstVP, iter);

            if(curdist > maxdist) {
                maxdist = curdist;
                secondVP.set(iter);
            }
        }

        return new DBIDVarTuple(firstVP, secondVP);
    }

    /**
     * First VP is selected by Maximum Variance
     * Second VP is selected by FFT
     * 
     * @param content Content to choose VP's from
     * @return Vantage Points
     */
    private DBIDVarTuple selectSampledMVFFTVantagePoints(DBIDs content) {
        DBIDVar firstVP = DBIDUtil.newVar();
        DBIDVar secondVP = DBIDUtil.newVar();
        DBIDVar currentDbid = DBIDUtil.newVar();

        // TODO: Truncate!
        if(content.size() == 1) {
            DBIDIter it = content.iter();
            firstVP.set(it);
            return new DBIDVarTuple(firstVP, secondVP);
        }

        // Create Workset to sample from
        final int adjustedSampleSize = Math.min(sampleSize, content.size());
        ArrayModifiableDBIDs scratchCopy = DBIDUtil.newArray(content);

        ModifiableDBIDs workset = DBIDUtil.randomSample(scratchCopy, adjustedSampleSize, random);

        double bestStandartDeviation = Double.NEGATIVE_INFINITY;
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
            }
        }

        double maxdist = 0;

        // Select second VP
        for(DBIDIter iter = workset.iter(); iter.valid(); iter.advance()) {
            double curdist = distance(firstVP, iter);

            if(curdist > maxdist) {
                maxdist = curdist;
                secondVP.set(iter);
            }
        }

        return new DBIDVarTuple(firstVP, secondVP);
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
         * "Left" Vantage point
         */
        DBIDRef firstVP;

        /**
         * "Right" Vantage point
         */
        DBIDRef secondVP;

        /**
         * child Trees
         */
        Node firstChild, secondChild;

        /**
         * upper and lower distance bounds
         */
        double firstLowBound, firstHighBound, secondLowBound, secondHighBound;

        /**
         * Constructor.
         * 
         */
        public Node() {
            this.firstLowBound = Double.MAX_VALUE;
            this.firstHighBound = -1;
            this.secondLowBound = Double.MAX_VALUE;
            this.secondHighBound = -1;
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
            DBIDRef firstVP = node.firstVP;
            DBIDRef secondVP = node.secondVP;


            double tau = knns.getKNNDistance();
            final double firstDistance = queryDistance(firstVP);
            knns.insert(firstDistance, firstVP);
            Node lc = node.firstChild;
            // TODO: Priortization?

            if(lc != null && node.firstLowBound <= firstDistance + tau && firstDistance - tau < node.firstHighBound) {
                tau = ghKNNSearch(knns, lc);
            }

            if(secondVP != null) {
                Node rc = node.secondChild;
                final double secondDistance = queryDistance(secondVP);
                knns.insert(secondDistance, secondVP);

                if(rc != null && node.secondLowBound <= secondDistance + tau && secondDistance - tau < node.secondHighBound) {
                    tau = ghKNNSearch(knns, rc);
                }
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
            final DBIDRef firstVP = node.firstVP;
            final DBIDRef secondVP = node.secondVP;

            final double firstVPDistance = queryDistance(firstVP);
            // TODO: if not null
            final double secondVPDistance = queryDistance(secondVP);

            if(firstVPDistance < range) {
                result.add(firstVPDistance, firstVP);
            }

            if(secondVPDistance < range) {
                result.add(secondVPDistance, secondVP);
            }

            if(secondVP != null) {

                Node lc = node.firstChild, rc = node.secondChild;

                final double firstDistanceDiff = (firstVPDistance - secondVPDistance) / 2;
                final double secondDistanceDiff = (secondVPDistance - firstDistanceDiff) / 2;

                // TODO: Bounds?
                if(lc != null && firstDistanceDiff < range) {
                    ghRangeSearch(result, lc, range);
                }

                if(rc != null && secondDistanceDiff < range) {
                    ghRangeSearch(result, rc, range);
                }
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
        private ComparableMinHeap<PrioritySearchBranch> heap = new ComparableMinHeap<>();

        /**
         * Stopping threshold.
         */
        private double threshold;

        /**
         * Current search position.
         */
        private PrioritySearchBranch cur;

        /**
         * Start the search.
         */
        public void doSearch() {
            this.threshold = Double.POSITIVE_INFINITY;
            this.heap.clear();
            this.heap.add(new PrioritySearchBranch(0, root, null));
            advance();
        }

        @Override
        public PrioritySearcher<Q> advance() {
            if(heap.isEmpty()) {
                cur = null;
                return this;
            }

            cur = heap.poll();

            if(cur.node != null) {
                double firstVPDist = queryDistance(cur.node.firstVP);
                // TODO if not null!
                double secondVPDist = queryDistance(cur.node.secondVP);
                Node lc = cur.node.firstChild;
                Node rc = cur.node.secondChild;

                if(lc != null && intersect(firstVPDist - threshold, firstVPDist + threshold, cur.node.firstLowBound, cur.node.firstHighBound)) {
                    final double mindist = Math.max(firstVPDist - cur.node.firstHighBound, cur.mindist);
                    heap.add(new PrioritySearchBranch(mindist, lc, DBIDUtil.deref(cur.node.firstVP)));
                }

                if(rc != null && intersect(secondVPDist - threshold, secondVPDist + threshold, cur.node.secondLowBound, cur.node.secondHighBound)) {
                    final double mindist = Math.max(secondVPDist - cur.node.secondHighBound, cur.mindist);
                    heap.add(new PrioritySearchBranch(mindist, rc, DBIDUtil.deref(cur.node.secondVP)));
                }
            }

            return this;

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
            return cur.vp.internalGetIndex();
        }

        @Override
        public boolean valid() {
            return cur != null;
        }

        @Override
        public PrioritySearcher<Q> decreaseCutoff(double threshold) {
            assert threshold <= this.threshold : "Thresholds must only decreasee.";
            this.threshold = threshold;
            return this;
        }

        @Override
        public double computeExactDistance() {
            return queryDistance(cur.vp);
        }

        @Override
        public double getLowerBound() {
            return cur.mindist;
        }

        @Override
        public double allLowerBound() {
            return cur.mindist;
        }
    }

    /**
     * Search position for priority search.
     *
     * @author Robert Gehde
     */
    private static class PrioritySearchBranch implements Comparable<PrioritySearchBranch> {
        /**
         * Minimum distance
         */
        double mindist;

        /**
         * associated vantage point
         */
        DBID vp;

        /**
         * Node
         */
        Node node;

        /**
         * Constructor.
         *
         * @param mindist Minimum distance
         * @param node Node
         * @param vp Vantage point
         */
        public PrioritySearchBranch(double mindist, Node node, DBID vp) {
            this.mindist = mindist;
            this.node = node;
            this.vp = vp;
        }

        @Override
        public int compareTo(PrioritySearchBranch o) {
            return Double.compare(this.mindist, o.mindist);
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
    public class GHTreePriorityDBIDSearcher extends GHTreePrioritySearcher<DBIDRef> {
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
            public final static OptionID MV_ALPHA_ID = new OptionID("ghtree.mvAlpha", "Threshold for Maximum Variance VP selection Algorithm");

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
                new DoubleParameter(MV_ALPHA_ID, 0.5) //
                        .addConstraint(CommonConstraints.LESS_EQUAL_ONE_DOUBLE).addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE).grab(config, x -> this.mvAlpha = x);
                new EnumParameter<>(VPSELECTOR_ID, VPSelectionAlgorithm.class).grab(config, x -> this.vpSelector = x);
            }

            @Override
            public Object make() {
                return new Factory<>(distance, random, sampleSize, truncate, mvAlpha, vpSelector);
            }
        }
    }
}
