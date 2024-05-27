package elki.index.tree.metrical.vptree;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;
import java.util.concurrent.TimeUnit;

import elki.data.NumberVector;
import elki.data.type.TypeInformation;
import elki.database.datastore.memory.MapIntegerDBIDDoubleStore;
import elki.database.ids.ArrayModifiableDBIDs;
import elki.database.ids.DBID;
import elki.database.ids.DBIDArrayIter;
import elki.database.ids.DBIDArrayMIter;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDMIter;
import elki.database.ids.DBIDRef;
import elki.database.ids.DBIDUtil;
import elki.database.ids.DBIDVar;
import elki.database.ids.DBIDs;
import elki.database.ids.DoubleDBIDHeap;
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

public class GHRPTree<O> implements DistancePriorityIndex<O> {

    /**
     * Class logger.
     */
    private static final Logging LOG = Logging.getLogger(GHRPTree.class);

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
    public GHRPTree(Relation<O> relation, Distance<? super O> distance, int leafsize, double mvAlpha, VPSelectionAlgorithm vpSelector) {
        this(relation, distance, RandomFactory.DEFAULT, leafsize, leafsize, mvAlpha, vpSelector);
    }

    /**
     * Constructor
     * 
     * @param relation data for tree construction
     * @param distance distance function for tree construction
     * @param random Random generator for sampling
     * @param sampleSize Sample size for finding the vantage point
     * @param truncate Leaf size threshold
     * @param mvAlpha Maximum Variance threshold
     * @param vpSelector Vantage Point selection Algorithm
     */
    public GHRPTree(Relation<O> relation, Distance<? super O> distance, RandomFactory random, int sampleSize, int truncate, double mvAlpha, VPSelectionAlgorithm vpSelector) {
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
        root = new Node(ReuseVPIndicator.ROOT);
        buildTree(root, relation.getDBIDs(), DBIDUtil.newVar(), DBIDUtil.newDistanceDBIDList());
        TreeParser parser = new TreeParser();
        parser.parseTree();
        System.gc();
        try {
            TimeUnit.SECONDS.sleep(2);
        }
        catch(InterruptedException e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
        System.gc();
    }

    private enum VPSelectionAlgorithm {
        RANDOM, FFT, MAXIMUM_VARIANCE, MAXIMUM_VARIANCE_SAMPLING, MAXIMUM_VARIANCE_FFT, MAXIMUM_VARIANCE_FFT_SAMPLING, REF_CHOOSE_VP
    }

    private enum ReuseVPIndicator {
        ROOT, FIRST_VP, SECOND_VP;
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
        ModifiableDoubleDBIDList firstVP;

        /**
         * "Right" Vantage point
         */
        ModifiableDoubleDBIDList secondVP;

        /**
         * Indicates which VP was reused.
         */
        ReuseVPIndicator vpIndicator;

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
        public Node(ReuseVPIndicator vpIndicator) {
            this.vpIndicator = vpIndicator;
            this.firstLowBound = Double.MAX_VALUE;
            this.firstHighBound = -1;
            this.secondLowBound = Double.MAX_VALUE;
            this.secondHighBound = -1;
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
     * @param reusePivot previous VP
     * @param secondDistances distances to previous VP
     * @return new node
     */
    private void buildTree(Node current, DBIDs content, DBIDRef reusePivot, ModifiableDoubleDBIDList reuseDistances) {
       
        // Check for truncate Leaf
        if(content.size() <= truncate) {
            DBIDIter contentIter = content.iter();
            DBID firstVantagePoint = DBIDUtil.deref(contentIter);
            ModifiableDoubleDBIDList vps = DBIDUtil.newDistanceDBIDList(content.size());

            vps.add(0, firstVantagePoint);

            double lowBound = Double.POSITIVE_INFINITY;
            double highBound = -1;

            for(contentIter.advance(); contentIter.valid(); contentIter.advance()) {
                double vpDist = distance(firstVantagePoint, contentIter);
                vps.add(distance(contentIter, firstVantagePoint), contentIter);

                if(vpDist != 0) {
                    lowBound = vpDist < lowBound ? vpDist : lowBound;
                }
                highBound = highBound < vpDist ? vpDist : highBound;
            }

            current.vpIndicator = ReuseVPIndicator.ROOT;
            current.firstVP = vps;
            current.firstLowBound = lowBound;
            current.firstHighBound = highBound;

            return;
        }
        DBIDVar firstVP = DBIDUtil.newVar();
        DBIDVar secondVP = DBIDUtil.newVar();
        ReuseVPIndicator vpIndicator = current.vpIndicator;

        switch(vpIndicator){
        case ROOT:
            DBIDVarTuple tuple = selectVPTuple(content);
            firstVP.set(tuple.first);
            secondVP.set(tuple.second);
            break;
        case FIRST_VP:
            firstVP.set(reusePivot);
            current.firstVP = DBIDUtil.newDistanceDBIDList();
            current.firstVP.add(0, reusePivot);
            secondVP.set(selectVPSingle(content, reusePivot));
            break;
        case SECOND_VP:
            secondVP.set(reusePivot);
            current.secondVP = DBIDUtil.newDistanceDBIDList();
            current.secondVP.add(0, reusePivot);
            firstVP.set(selectVPSingle(content, reusePivot));
            break;
        }

        assert !DBIDUtil.equal(firstVP, secondVP);
        assert firstVP != null && !firstVP.isEmpty() && firstVP.isSet();

        // Count objects tied with first vp
        int tiedFirst = 0;

        for(DBIDIter contentIter = content.iter(); contentIter.valid(); contentIter.advance()) {
            if(DBIDUtil.equal(firstVP, contentIter)) {
                tiedFirst++;
            }
        }

        // many duplicates of first Vantage Point
        if(tiedFirst + truncate > content.size()) {
            ModifiableDoubleDBIDList vps = DBIDUtil.newDistanceDBIDList(content.size());
            double lowBound = Double.POSITIVE_INFINITY;
            double highBound = -1;

            for(DBIDIter contentIter = content.iter(); contentIter.valid(); contentIter.advance()) {
                double vpDist = distance(firstVP, contentIter);
                vps.add(distance(contentIter, firstVP), contentIter);

                if(vpDist != 0) {
                    lowBound = vpDist < lowBound ? vpDist : lowBound;
                }
                highBound = highBound < vpDist ? vpDist : highBound;
            }

            current.vpIndicator = ReuseVPIndicator.ROOT;
            current.firstVP = vps;
            current.firstLowBound = lowBound;
            current.firstHighBound = highBound;
            return;
        }
        
        // If second VP is empty, Leaf is reached, just set low/highbound
        // Else build childnodes
        if(secondVP == null || secondVP.isEmpty() || !secondVP.isSet()) {
            current.firstVP = DBIDUtil.newDistanceDBIDList(tiedFirst);
            for(DBIDIter contentIter = content.iter(); contentIter.valid(); contentIter.advance()) {
                current.firstVP.add(0, contentIter);
            }
            current.firstLowBound = 0;
            current.firstHighBound = Double.POSITIVE_INFINITY;
        }
        else {
            // TODO: Wat?
            if(current.firstVP == null){
                current.firstVP = DBIDUtil.newDistanceDBIDList(tiedFirst);
            }

            // TODO: Wat?
            if (current.secondVP == null || current.secondVP.isEmpty()){
                // count tied to second vp
                int tiedSecond = 0;
                for(DBIDIter contentIter = content.iter(); contentIter.valid(); contentIter.advance()) {
                    if(DBIDUtil.equal(secondVP, contentIter)) {
                        tiedSecond++;
                    }
                }
                current.secondVP = DBIDUtil.newDistanceDBIDList(tiedSecond);
            }

            ArrayModifiableDBIDs contentArray = DBIDUtil.newArray(content);

            double firstDistance;
            double secondDistance;

            ModifiableDoubleDBIDList firstDistances = DBIDUtil.newDistanceDBIDList();
            ModifiableDoubleDBIDList secondDistances = DBIDUtil.newDistanceDBIDList();

            ModifiableDBIDs firstChildren = null;
            ModifiableDBIDs secondChildren = null;

            for(DBIDArrayMIter iter = contentArray.iter(); iter.valid(); iter.advance()) {
                // If the current position is a VP, this will be set to current
                // offset
                int vpOffset = -1;

                // TODO: add binary Decision to GH
                switch(vpIndicator){
                case FIRST_VP:
                    firstDistance = reuseDistances.doubleValue(iter.getOffset());
                    secondDistance = distance(secondVP, iter);
                    break;
                case SECOND_VP:
                    firstDistance = distance(firstVP, iter);
                    secondDistance = reuseDistances.doubleValue(iter.getOffset());
                    break;
                default:
                    firstDistance = distance(firstVP, iter);
                    secondDistance = distance(secondVP, iter);
                    break;
                }

                if(DBIDUtil.equal(firstVP, iter) || firstDistance == 0) {
                    vpOffset = 0;
                    current.firstVP.add(firstDistance, iter);
                }

                if(DBIDUtil.equal(secondVP, iter) || secondDistance == 0) {
                    vpOffset = 1;
                    current.secondVP.add(secondDistance, iter);
                }

                if(vpOffset == -1) {
                    if(firstDistance < secondDistance) {
                        firstDistances.add(firstDistance, iter);
                        if(firstChildren == null) {
                            firstChildren = DBIDUtil.newArray();
                        }
                        firstChildren.add(iter);
                    }
                    else {
                        secondDistances.add(secondDistance, iter);
                        if(secondChildren == null) {
                            secondChildren = DBIDUtil.newArray();
                        }
                        secondChildren.add(iter);
                    }
                }

                if(vpOffset != 0){
                    current.firstLowBound = current.firstLowBound > firstDistance ? firstDistance : current.firstLowBound;
                }
                current.firstHighBound = current.firstHighBound < firstDistance ? firstDistance : current.firstHighBound;
                
                if(vpOffset != 1){
                    current.secondLowBound = current.secondLowBound > secondDistance ? secondDistance : current.secondLowBound;
                }
                current.secondHighBound = current.secondHighBound < secondDistance ? secondDistance : current.secondHighBound;
            }

            if(firstChildren != null) {
                buildTree(current.firstChild = new Node(ReuseVPIndicator.FIRST_VP), firstChildren, firstVP, firstDistances);
            }

            if(secondChildren != null) {
                buildTree(current.secondChild = new Node(ReuseVPIndicator.SECOND_VP), secondChildren, secondVP, secondDistances);
            }
        }
    }

    private DBIDVarTuple selectVPTuple(DBIDs content) {
        DBIDVarTuple tuple;

        switch(this.vpSelector){
        case RANDOM:
            tuple = selectRandomVantagePoints(content);
            break;
        case FFT:
            tuple = selectFFTVantagePoints(content);
            break;
        case MAXIMUM_VARIANCE:
            tuple = selectMaximumVarianceVantagePoints(content);
            break;
        case MAXIMUM_VARIANCE_SAMPLING:
            tuple = selectSampledMaximumVarianceVantagePoints(content);
            break;
        case MAXIMUM_VARIANCE_FFT:
            tuple = selectMVFFTVantagePoints(content);
            break;
        case MAXIMUM_VARIANCE_FFT_SAMPLING:
            tuple = selectSampledMVFFTVantagePoints(content);
            break;
        default:
            tuple = selectFFTVantagePoints(content);
            break;
        }

        return tuple;
    }

    /**
     * Selects a single Vantage Point in the context of e given (reused) first
     * VP
     * 
     * @param content the Set to choose VP from
     * @param vantagePoint the reused Vantage Point
     * @return Vantage Point
     */
    private DBIDVar selectVPSingle(DBIDs content, DBIDRef vantagePoint) {

        switch(this.vpSelector){
        case RANDOM:
            return selectSingleRandomVantagePoint(content);
        case FFT:
            return selectSecondFFTVantagePoint(vantagePoint, content);
        case MAXIMUM_VARIANCE:
            return selectSingleMVVP(content);
        case MAXIMUM_VARIANCE_SAMPLING:
            return selectSampledSingleMVVP(content);
        case MAXIMUM_VARIANCE_FFT:
            return selectSecondFFTVantagePoint(vantagePoint, content);
        case MAXIMUM_VARIANCE_FFT_SAMPLING:
            return selectSecondFFTVantagePoint(vantagePoint, content);
        default:
            return selectSecondFFTVantagePoint(vantagePoint, content);
        }

    }

    /**
     * Finds two Vantage Points using Farthest first Traversal
     * 
     * @param content Content to choose VP's from
     * @return two Vantage points
     */
    private DBIDVarTuple selectFFTVantagePoints(DBIDs content) {
        // First VP = random
        DBIDVar first = selectSingleRandomVantagePoint(content);

        // Second VP = farthest away from first
        DBIDVar second = selectSecondFFTVantagePoint(first, content);

        return new DBIDVarTuple(first, second);
    }

    /**
     * Selects a single Random Vantage Point from Content
     * 
     * @param content the Set to choose a Vantage Point from
     * @return Vantage Point
     */
    private DBIDVar selectSingleRandomVantagePoint(DBIDs content) {
        ArrayModifiableDBIDs contentArray = DBIDUtil.newArray(content);
        DBIDArrayIter contentIter = contentArray.iter();
        DBIDVar first = DBIDUtil.newVar();

        if(content.size() == 1){
            first.set(contentIter);
            return first;
        }

        int pos = randomThread.nextInt(content.size());

        contentIter.seek(pos);
        first.set(contentIter);

        return first;
    }

    /***
     * Selects a Vantage Point from content farthest away from the first one.
     * 
     * @param firstVP the first Vantage Point
     * @param content the Set to choose a second Vantage Point from
     * @return a Vantage Point
     */
    private DBIDVar selectSecondFFTVantagePoint(DBIDRef firstVP, DBIDs content) {
        // Second VP = farthest away from first
        DBIDVar second = DBIDUtil.newVar();
        double maxDist = 0;

        ArrayModifiableDBIDs contentArray = DBIDUtil.newArray(content);
        DBIDArrayIter contentIter = contentArray.iter();
        for(contentIter.seek(0); contentIter.valid(); contentIter.advance()) {
            double distance = distance(firstVP, contentIter);
            if(distance > maxDist) {
                second.set(contentIter);
                maxDist = distance;
            }
        }

        return second;
    }

    /**
     * Returns two random Vantage Points. Ignores duplicates.
     * 
     * @param content Content to choose VP's from
     * @return two Vantage points
     */
    private DBIDVarTuple selectRandomVantagePoints(DBIDs content) {
        // First VP = random
        DBIDVar first = selectSingleRandomVantagePoint(content);

        DBIDVar second = DBIDUtil.newVar();
        // Set the first DBID nonequal to first vp as second vp
        for(DBIDIter it = content.iter(); it.valid(); it.advance()) {
            if(!DBIDUtil.equal(it, first) && !(distance(it, first) == 0)) {
                second.set(it);
                assert distance(first, second) > 0;
            }
        }

        return new DBIDVarTuple(first, second);
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

        if (content.size() == 1){
            DBIDIter it = content.iter();
            firstVP.set(it);
            return new DBIDVarTuple(firstVP, secondVP);
        }

        MapIntegerDBIDDoubleStore means = new MapIntegerDBIDDoubleStore(content.size());
        DoubleDBIDHeap stds = DBIDUtil.newMaxHeap(content.size());
        double bestMean = 0;
        double maxDist = 0;

        // Calculate means and stds
        for(DBIDIter it = content.iter(); it.valid(); it.advance()) {
            currentDbid.set(it);

            MeanVariance currentVariance = new MeanVariance();

            for(DBIDIter jt = content.iter(); jt.valid(); jt.advance()) {
                double currentDistance = distance(currentDbid, jt);

                currentVariance.put(currentDistance);

                if(currentDistance > maxDist) {
                    maxDist = currentDistance;
                }
            }

            double currentMean = currentVariance.getMean();
            double currentStandartDeviance = currentVariance.getSampleStddev();

            if(currentMean > bestMean){
                bestMean = currentMean;
            }

            means.put(currentDbid,currentMean);
            stds.insert(currentStandartDeviance, currentDbid);
        }

        double omega = this.mvAlpha * maxDist;
        
        while(!stds.isEmpty() && (!secondVP.isSet() || !firstVP.isSet())) {
            DBIDRef currentDBID = stds;

            if(!firstVP.isSet()) {
                firstVP.set(currentDBID);
                // Only duplicates in content
                if(bestMean == 0) {
                    return new DBIDVarTuple(firstVP, secondVP);
                }
            }
            else {
                double firstVPDist = distance(firstVP, stds);
                if(!DBIDUtil.equal(stds, firstVP) && Math.abs(firstVPDist - bestMean) <= omega && firstVPDist != 0) {
                    secondVP.set(currentDBID);
                }
            }
            stds.poll();
        }
        return new DBIDVarTuple(firstVP, secondVP);
    }

    /**
     * selects only the single Vantage Point with maximum Variance
     * with no respect to a second VP
     * 
     * @param content the set to choose the VP from
     * @return Vantage Point
     */
    private DBIDVar selectSingleMVVP(DBIDs content) {
        DBIDVar firstVP = DBIDUtil.newVar();
        DBIDVar currentDbid = DBIDUtil.newVar();

        // TODO: Truncate!
        if(content.size() == 1) {
            DBIDIter it = content.iter();
            firstVP.set(it);
            return firstVP;
        }

        double bestStandartDeviation = Double.NEGATIVE_INFINITY;
        // Modifiable copy for selecting:
        ArrayModifiableDBIDs workset = DBIDUtil.newArray(content);

        // Select first VP
        for(DBIDMIter it = workset.iter(); it.valid(); it.advance()) {
            currentDbid.set(it);

            MeanVariance currentVariance = new MeanVariance();

            for(DBIDMIter jt = workset.iter(); jt.valid(); jt.advance()) {
                double currentDistance = distance(currentDbid, jt);

                currentVariance.put(currentDistance);
            }

            double currentStandartDeviance = currentVariance.getSampleStddev();

            if(currentStandartDeviance > bestStandartDeviation) {
                firstVP.set(it);
                bestStandartDeviation = currentStandartDeviance;
            }
        }

        return firstVP;
    }

    /**
     * sampled version of single Select MV Vantage Point
     * 
     * @param content the Set to select VP from
     * @return Vantage Point
     */
    private DBIDVar selectSampledSingleMVVP(DBIDs content) {
        DBIDVar result = DBIDUtil.newVar();

        if (this.sampleSize == 1 ){
            return this.selectSingleRandomVantagePoint(content);
        }

        if(content.size() == 1) {
            DBIDIter it = content.iter();
            result.set(it);
            return result;
        }

        // Create Workset to sample from
        final int adjustedSampleSize = Math.min(sampleSize, content.size());
        ArrayModifiableDBIDs contentCopy = DBIDUtil.newArray(content);
        ModifiableDBIDs workset = DBIDUtil.randomSample(contentCopy, adjustedSampleSize, random);

        return selectSingleMVVP(workset);
    }

    /**
     * Select Maximum Variance Vantage Points using a random sampled subset
     * of relation
     * 
     * @param content Content to choose VP's from
     * @return Vantage Points
     */
    private DBIDVarTuple selectSampledMaximumVarianceVantagePoints(DBIDs content) {
        if(this.sampleSize == 2 || this.sampleSize == 1) {
            return this.selectRandomVantagePoints(content);
        }

        // Create Workset to sample from
        final int adjustedSampleSize = Math.min(sampleSize, content.size());
        ArrayModifiableDBIDs contentCopy = DBIDUtil.newArray(content);

        ModifiableDBIDs workset = DBIDUtil.randomSample(contentCopy, adjustedSampleSize, random);

        return selectMaximumVarianceVantagePoints(workset);
    }

    /**
     * First VP is selected by Maximum Variance
     * Second VP is selected by FFT
     * 
     * @param content Content to choose VP's from
     * @return Vantage Points
     */
    private DBIDVarTuple selectMVFFTVantagePoints(DBIDs content) {
        DBIDVar firstVP = DBIDUtil.newVar();
        DBIDVar secondVP = DBIDUtil.newVar();

        if(content.size() == 1) {
            DBIDIter iter = content.iter();
            firstVP.set(iter);
            return new DBIDVarTuple(firstVP, secondVP);
        }

        // Select first VP
        firstVP = selectSingleMVVP(content);

        // Select second VP
        secondVP = selectSecondFFTVantagePoint(firstVP, content);

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

        if(content.size() == 1) {
            DBIDIter it = content.iter();
            firstVP.set(it);
            return new DBIDVarTuple(firstVP, secondVP);
        }

        // Create Workset to sample from
        final int adjustedSampleSize = Math.min(sampleSize, content.size());
        ArrayModifiableDBIDs contentCopy = DBIDUtil.newArray(content);
        ModifiableDBIDs workset = DBIDUtil.randomSample(contentCopy, adjustedSampleSize, random);

        // Select first VP
        firstVP = selectSingleMVVP(workset);

        // Select second VP
        secondVP = selectSecondFFTVantagePoint(firstVP, content);
        return new DBIDVarTuple(firstVP, secondVP);
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
    public void logStatistics() {
        LOG.statistics(new LongStatistic(this.getClass().getName() + ".distance-computations", distComputations));
    }

    @Override
    public KNNSearcher<O> kNNByObject(DistanceQuery<O> distanceQuery, int maxk, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHRPTreeKNNObjectSearcher() : null;
    }

    @Override
    public KNNSearcher<DBIDRef> kNNByDBID(DistanceQuery<O> distanceQuery, int maxk, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHRPTreeKNNDBIDSearcher() : null;
    }

    @Override
    public RangeSearcher<O> rangeByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHRPTreeRangeObjectSearcher() : null;
    }

    @Override
    public RangeSearcher<DBIDRef> rangeByDBID(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHRPTreeRangeDBIDSearcher() : null;
    }

    @Override
    public PrioritySearcher<O> priorityByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHRPTreePriorityObjectSearcher() : null;
    }

    @Override
    public PrioritySearcher<DBIDRef> priorityByDBID(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHRPTreePriorityDBIDSearcher() : null;
    }

    /**
     * kNN search for the GHRP-Tree
     * 
     * @author Sebastian Aloisi
     */
    public static abstract class GHRPTreeKNNSearcher {
        /**
         * Recursive search function
         * 
         * @param knns Current kNN results
         * @param node Current node
         * @return New tau
         */
        protected double ghrpKNNSearch(KNNHeap knns, Node node, double reusedDistance) {
            ReuseVPIndicator vpIndicator = node.vpIndicator;

            final DBIDs firstVP = node.firstVP;
            final DBIDs secondVP = node.secondVP;

            double tau = knns.getKNNDistance();

            double firstDistance = 0;

            if(vpIndicator == ReuseVPIndicator.FIRST_VP) {
                firstDistance = reusedDistance;
            }
            else {
                // Check knn for vp and singletons
                for(DBIDIter firstVPiter = firstVP.iter(); firstVPiter.valid(); firstVPiter.advance()) {
                    firstDistance = queryDistance(firstVPiter);
                    knns.insert(firstDistance, firstVPiter);
                }
            }

            tau = knns.getKNNDistance();

            if(secondVP != null && !secondVP.isEmpty()) {
                Node lc = node.firstChild;
                Node rc = node.secondChild;

                double secondDistance  = 0;

                if(vpIndicator == ReuseVPIndicator.SECOND_VP) {
                    secondDistance = reusedDistance;
                }
                else {
                    for(DBIDIter secondVPIter = secondVP.iter(); secondVPIter.valid(); secondVPIter.advance()) {
                        // TODO: only query once?
                        secondDistance = queryDistance(secondVPIter);
                        knns.insert(secondDistance, secondVPIter);
                    }
                }

                tau = knns.getKNNDistance();

                final double firstDistanceDiff = (firstDistance - secondDistance) / 2;
                final double secondDistanceDiff = (secondDistance - firstDistance) / 2;

                // TODO: less or equal less?
                if(firstDistance < 0) {
                    if(lc != null && firstDistanceDiff < tau && node.firstLowBound <= firstDistance + tau && firstDistance - tau <= node.firstHighBound) {
                        tau = ghrpKNNSearch(knns, lc, firstDistance);
                    }

                    if(rc != null && secondDistanceDiff < tau && node.secondLowBound <= secondDistance + tau && secondDistance - tau <= node.secondHighBound) {
                        tau = ghrpKNNSearch(knns, rc, secondDistance);
                    }
                }
                else {
                    if(rc != null && secondDistanceDiff < tau && node.secondLowBound <= secondDistance + tau && secondDistance - tau <= node.secondHighBound) {
                        tau = ghrpKNNSearch(knns, rc, secondDistance);
                    }
                    if(lc != null && firstDistanceDiff < tau && node.firstLowBound <= firstDistance + tau && firstDistance - tau <= node.firstHighBound) {
                        tau = ghrpKNNSearch(knns, lc, firstDistance);
                    }

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

    public class GHRPTreeKNNObjectSearcher extends GHRPTreeKNNSearcher implements KNNSearcher<O> {
        private O query;

        @Override
        public KNNList getKNN(O query, int k) {
            final KNNHeap knns = DBIDUtil.newHeap(k);
            this.query = query;
            ghrpKNNSearch(knns, root, 0);
            return knns.toKNNList();
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHRPTree.this.distance(query, p);
        }
    }

    /**
     * 
     * kNN search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     */
    public class GHRPTreeKNNDBIDSearcher extends GHRPTreeKNNSearcher implements KNNSearcher<DBIDRef> {

        private DBIDRef query;

        @Override
        public KNNList getKNN(DBIDRef query, int k) {
            final KNNHeap knns = DBIDUtil.newHeap(k);
            this.query = query;
            ghrpKNNSearch(knns, root, 0);
            return knns.toKNNList();
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHRPTree.this.distance(query, p);
        }
    }

    /**
     * Range Searcher for the GHRP-Tree
     * 
     * @author Sebastian Aloisi
     */
    public static abstract class GHRPTreeRangeSearcher {

        protected void ghrpRangeSearch(ModifiableDoubleDBIDList result, Node node, double reusedDistance, double range) {
            ReuseVPIndicator vpIndicator = node.vpIndicator;
            final DBIDs firstVP = node.firstVP;
            final DBIDs secondVP = node.secondVP;

            double firstVPDistance = 0;
            if(vpIndicator == ReuseVPIndicator.FIRST_VP) {
                firstVPDistance = reusedDistance;
            }
            else {
                for(DBIDIter firstVPIter = firstVP.iter(); firstVPIter.valid(); firstVPIter.advance()) {
                    firstVPDistance = queryDistance(firstVPIter);

                    if(firstVPDistance <= range) {
                        result.add(firstVPDistance, firstVPIter);
                    }
                }
            }

            if(secondVP != null) {
                double secondVPDistance = 0;

                if(vpIndicator == ReuseVPIndicator.SECOND_VP) {
                    secondVPDistance = reusedDistance;
                }
                else {
                    for(DBIDIter secondVPIter = secondVP.iter(); secondVPIter.valid(); secondVPIter.advance()) {
                        secondVPDistance = queryDistance(secondVPIter);
                        
                        if(secondVPDistance <= range) {
                            result.add(secondVPDistance, secondVPIter);
                        }
                    }
                }

                Node lc = node.firstChild, rc = node.secondChild;

                final double firstDistanceDiff = (firstVPDistance - secondVPDistance) / 2;
                final double secondDistanceDiff = (secondVPDistance - firstVPDistance) / 2;

                if(lc != null && firstDistanceDiff < range && node.firstLowBound <= firstVPDistance + range && firstVPDistance - range <= node.firstHighBound) {
                    ghrpRangeSearch(result, lc, range, firstVPDistance);
                }

                if(rc != null && secondDistanceDiff < range && node.secondLowBound <= secondVPDistance + range && secondVPDistance - range <= node.secondHighBound) {
                    ghrpRangeSearch(result, rc, range, secondVPDistance);
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
     * Range search for the GHRP-Tree.
     * 
     * @author Sebastian Aloisi
     */
    public class GHRPTreeRangeObjectSearcher extends GHRPTreeRangeSearcher implements RangeSearcher<O> {

        private O query;

        @Override
        public ModifiableDoubleDBIDList getRange(O query, double range, ModifiableDoubleDBIDList result) {
            this.query = query;
            ghrpRangeSearch(result, root, 0, range);
            return result;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHRPTree.this.distance(query, p);
        }
    }

    /**
     * Range search for the GHRP-Tree
     * 
     * @author Sebastian Aloisi
     */
    public class GHRPTreeRangeDBIDSearcher extends GHRPTreeRangeSearcher implements RangeSearcher<DBIDRef> {

        private DBIDRef query;

        @Override
        public ModifiableDoubleDBIDList getRange(DBIDRef query, double range, ModifiableDoubleDBIDList result) {
            this.query = query;
            ghrpRangeSearch(result, root, 0, range);
            return result;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHRPTree.this.distance(query, p);
        }
    }

    /**
     * Priority search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     * 
     * @param <Q> query type
     */
    public abstract class GHRPTreePrioritySearcher<Q> implements PrioritySearcher<Q> {
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

        // TODO: adjust to reusement of VP?
        @Override
        public PrioritySearcher<Q> advance() {
            if(heap.isEmpty()) {
                cur = null;
                return this;
            }

            cur = heap.poll();

            if(cur.node != null) {
                DBIDs firstVP = cur.node.firstVP;
                Node lc = cur.node.firstChild;

                // TODO: loop necessary?
                for(DBIDIter firstVPIter = firstVP.iter(); firstVPIter.valid(); firstVPIter.advance()) {
                    double firstVPDist = queryDistance(firstVPIter);
                    if(lc != null && intersect(firstVPDist - threshold, firstVPDist + threshold, cur.node.firstLowBound, cur.node.firstHighBound)) {
                        final double mindist = Math.max(firstVPDist - cur.node.firstHighBound, cur.mindist);
                        heap.add(new PrioritySearchBranch(mindist, lc, DBIDUtil.deref(firstVPIter)));
                    }
                }

                if(cur.node.secondVP != null) {
                    DBIDs secondVP = cur.node.secondVP;
                    DBIDIter secondVPIter = secondVP.iter();
                    double secondVPDist = queryDistance(secondVPIter);
                    Node rc = cur.node.secondChild;

                    if(rc != null && intersect(secondVPDist - threshold, secondVPDist + threshold, cur.node.secondLowBound, cur.node.secondHighBound)) {
                        final double mindist = Math.max(secondVPDist - cur.node.secondHighBound, cur.mindist);
                        heap.add(new PrioritySearchBranch(mindist, rc, DBIDUtil.deref(secondVPIter)));
                    }

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
     * Priority Range Search for the GHRP-tree
     * 
     * @author Sebastian Aloisi
     * 
     */
    public class GHRPTreePriorityObjectSearcher extends GHRPTreePrioritySearcher<O> {
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
            return GHRPTree.this.distance(query, p);
        }
    }

    /**
     * Priority Range Search for the GHRP-tree
     * 
     * @author Sebastian Aloisi
     * 
     */
    public class GHRPTreePriorityDBIDSearcher extends GHRPTreePrioritySearcher<DBIDRef> {
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
            return GHRPTree.this.distance(query, p);
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
     * Index factory for the GHrp-Tree
     * 
     * @author Sebastian Aloisi
     * 
     * @param <O> Object type
     */
    @Alias({ "ghrp" })
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
            return new GHRPTree<>(relation, distance, random, sampleSize, truncate, mvAlpha, vpSelector);
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
            public final static OptionID DISTANCE_FUNCTION_ID = new OptionID("ghrptree.distanceFunction", "Distance function to determine the distance between objects");

            /**
             * Parameter to specify the sample size for choosing vantage points
             */
            public final static OptionID SAMPLE_SIZE_ID = new OptionID("ghrptree.sampleSize", "Size of sample to select vantage points from");

            /**
             * Parameter to specify the minimum leaf size
             */
            public final static OptionID TRUNCATE_ID = new OptionID("ghrptree.truncate", "Minimum leaf size for stopping");

            /**
             * Parameter to specify Maximum Variance Threshold
             */
            public final static OptionID MV_ALPHA_ID = new OptionID("ghrptree.mvAlpha", "Threshold for Maximum Variance VP selection Algorithm");

            /**
             * Parameter to specify the rnd generator seed
             */
            public final static OptionID SEED_ID = new OptionID("ghrptree.seed", "The rnd number generator seed");

            /**
             * Parameter to specify the Vantage Point selection Algorithm
             */
            public final static OptionID VPSELECTOR_ID = new OptionID("ghrptree.vpSelector", "The Vantage Point selection Algorithm");

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
                                LOG.warning("GHRPTree requires a metric to be exact.");
                            }
                        });
                new IntParameter(SAMPLE_SIZE_ID, 10) //
                        .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
                        .grab(config, x -> this.sampleSize = x);
                new IntParameter(TRUNCATE_ID, 8) //
                        .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
                        .grab(config, x -> this.truncate = x);
                new RandomParameter(SEED_ID).grab(config, x -> random = x);
                new DoubleParameter(MV_ALPHA_ID, 0.15) //
                        .addConstraint(CommonConstraints.LESS_EQUAL_ONE_DOUBLE).addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE).grab(config, x -> this.mvAlpha = x);
                new EnumParameter<>(VPSELECTOR_ID, VPSelectionAlgorithm.class).grab(config, x -> this.vpSelector = x);
            }

            @Override
            public Object make() {
                return new Factory<>(distance, random, sampleSize, truncate, mvAlpha, vpSelector);
            }
        }
    }

    private class TreeParser {
        private LinkedList<String> nodes;

        private LinkedList<String> edges;

        private int objectCounter;

        private DecimalFormat decimalFormat;

        public TreeParser() {
            this.nodes = new LinkedList<String>();
            this.edges = new LinkedList<String>();
            this.objectCounter = 0;
            this.decimalFormat = new DecimalFormat("0.00");
        }

        public void parseTree() {
            parseNode(root);
            String treeString = treeToString();
            try {
                FileWriter fileWriter = new FileWriter("ghrp.dot");
                fileWriter.write(treeString);
                fileWriter.close();
            }
            catch(IOException e) {

            }
        }

        private void parseNode(Node node) {
            DBIDs firstVP = node.firstVP;
            DBIDs secondVP = node.secondVP;
            ReuseVPIndicator vpIndicator = node.vpIndicator;

            int objectsInNode = 0;

            if (vpIndicator == ReuseVPIndicator.ROOT || vpIndicator != ReuseVPIndicator.FIRST_VP){
                objectsInNode += firstVP.size();
            }


            if (node.secondVP != null){
                if(vpIndicator == ReuseVPIndicator.ROOT || vpIndicator != ReuseVPIndicator.SECOND_VP) {
                    objectsInNode += secondVP.size();
                }
            }

            String firstLowBound = String.valueOf(this.decimalFormat.format(node.firstLowBound));
            String firstHighBound = String.valueOf(this.decimalFormat.format(node.firstHighBound));


            String nodeID = getNodeID(node);

            String secondLowBound = node.secondLowBound == Double.MAX_VALUE ? "MAX_VAL" : String.valueOf(this.decimalFormat.format(node.secondLowBound));
            String secondHighBound = String.valueOf(this.decimalFormat.format(node.secondHighBound));

            String nodeString = "\"" +nodeID + "\" [ label = \"ID: " + nodeID + "\\n sID: \\n obj: " + String.valueOf(objectsInNode) + "\\n flb: " + firstLowBound + "\\n fhb: " + firstHighBound + "\\n slb: " + secondLowBound + "\\n shb: " + secondHighBound + "\"]\n";
            this.nodes.add(nodeString);
            this.objectCounter += objectsInNode;

            Node firstChild = node.firstChild;
            Node secondChild = node.secondChild;

            if(firstChild != null) {
                String firstChildID = getNodeID(firstChild);
                this.edges.add("\"" + nodeID + "\" -> \"" + firstChildID + "\"\n");
                parseNode(firstChild);
            }

            if(secondChild != null) {
                String secondChildID = getNodeID(secondChild);
                this.edges.add("\"" + nodeID + "\" -> \"" + secondChildID + "\"\n");
                parseNode(secondChild);
            }
        }

        private String getNodeID (Node node){
            DBIDIter firstVPIter = node.firstVP.iter();
            DBIDIter secondVPIter;
            String firstVPID = String.valueOf(firstVPIter.internalGetIndex());
            String secondVPID = "NaN";
            if(node.secondVP != null) {
                secondVPIter = node.secondVP.iter();
                secondVPID = node.secondVP.isEmpty() ? "NAN" : String.valueOf(secondVPIter.internalGetIndex());
            }
            return firstVPID + "<>" + secondVPID;
        }

        private String treeToString() {
            String header = "digraph {\nrankdir=\"TB\"\nnode [shape=box]\n";
            String stats = "stats [label=\"Objects found: " + this.objectCounter + "\"]\n";
            String tail = "}";
            StringBuilder bodyStringBuilder = new StringBuilder();
            String body, result;

            Iterator nodeIter = this.nodes.iterator();

            while(nodeIter.hasNext()) {
                bodyStringBuilder.append(nodeIter.next());
            }

            Iterator edgesIterator = this.edges.iterator();

            while(edgesIterator.hasNext()) {
                bodyStringBuilder.append(edgesIterator.next());
            }

            body = bodyStringBuilder.toString();

            result = header + stats + body + tail;

            return result;
        }
    }
}
