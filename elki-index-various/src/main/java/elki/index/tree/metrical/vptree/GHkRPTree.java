package elki.index.tree.metrical.vptree;

import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Random;

import elki.data.NumberVector;
import elki.data.type.TypeInformation;
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
 * Generalized hyperplane Tree using k-fold splits
 * <p>
 * Uses two Vantage Points for Generalized Hyperplane decomposition
 * and k-fold splits to partition the relation.
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
public class GHkRPTree<O> implements DistancePriorityIndex<O> {
    /**
     * Class logger.
     */
    private static final Logging LOG = Logging.getLogger(GHTree.class);

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
     * The k Value to split to
     */
    static int kVal;

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
    public GHkRPTree(Relation<O> relation, Distance<? super O> distance, int leafsize, int kVal, double mvAlpha, VPSelectionAlgorithm vpSelector) {
        this(relation, distance, RandomFactory.DEFAULT, leafsize, leafsize, kVal, mvAlpha, vpSelector);
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
    public GHkRPTree(Relation<O> relation, Distance<? super O> distance, RandomFactory random, int sampleSize, int truncate, int kVal, double mvAlpha, VPSelectionAlgorithm vpSelector) {
        this.relation = relation;
        this.distFunc = distance;
        this.random = random;
        this.randomThread = random.getSingleThreadedRandom();
        this.distQuery = distance.instantiate(relation);
        this.sampleSize = Math.max(sampleSize, 1);
        this.truncate = Math.max(truncate, 1);
        this.kVal = kVal;
        this.mvAlpha = mvAlpha;
        this.vpSelector = vpSelector;
    }

    @Override
    public void initialize() {
        root = new Node(this.kVal, ReuseVPIndicator.ROOT);
        buildTree(root, relation.getDBIDs(), DBIDUtil.newVar(), DBIDUtil.newDistanceDBIDList());
        TreeParser treeParser = new TreeParser();
        treeParser.parseTree();
    }

    private enum VPSelectionAlgorithm {
        RANDOM, FFT, MAXIMUM_VARIANCE, MAXIMUM_VARIANCE_SAMPLING, MAXIMUM_VARIANCE_FFT, MAXIMUM_VARIANCE_FFT_SAMPLING, REF_CHOOSE_VP
    }

    private enum ReuseVPIndicator {
        ROOT, FIRST_VP, SECOND_VP;
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
     * @return new node
     */
    private void buildTree(Node current, DBIDs content, DBIDRef reusePivot, ModifiableDoubleDBIDList reuseDistances) {
        
        // Check for truncate Leaf
        if(content.size() <= truncate) {
            DBIDIter contentIter = content.iter();
            DBID firstVantagePoint = DBIDUtil.deref(contentIter);
            ModifiableDoubleDBIDList vps = DBIDUtil.newDistanceDBIDList(content.size());

            vps.add(0, firstVantagePoint);

            for(contentIter.advance(); contentIter.valid(); contentIter.advance()) {
                vps.add(distance(contentIter, firstVantagePoint), contentIter);
            }

            // Set Indicator to Root, otherwise this node may not be searchable
            current.vpIndicator = ReuseVPIndicator.ROOT;

            current.firstVP = vps;
            current.secondVP = null;
            current.childNodes = null;
            for(int i = 0; i < current.kVal ; i++){
                current.firstLowerBounds[i] = 0;
                current.secondLowerBounds[i] = 0;
                current.firstUpperBounds[i] = 0;
                current.secondUpperBounds[i] = 0;
            }

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
        assert !firstVP.isEmpty() || !firstVP.isSet();

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
            for(DBIDIter contentIter = content.iter(); contentIter.valid(); contentIter.advance()) {
                vps.add(distance(contentIter, firstVP), contentIter);
            }

            current.firstVP = vps;
            current.secondVP = null;
            current.childNodes = null;
            for(int i = 0; i < current.kVal; i++) {
                current.firstLowerBounds[i] = 0;
                current.secondLowerBounds[i] = 0;
                current.firstUpperBounds[i] = 0;
                current.secondUpperBounds[i] = 0;
            }

            return;
        }

        // If second VP is empty, Leaf is reached, just set low/highbound
        // Else build childnodes
        if(secondVP == null || secondVP.isEmpty() || !secondVP.isSet()) {
            current.firstVP = DBIDUtil.newDistanceDBIDList(tiedFirst);
            for(DBIDIter contentIter = content.iter(); contentIter.valid(); contentIter.advance()) {
                current.firstVP.add(0, contentIter);
            }
            for (int i = 0; i < kVal ; i++){
                current.firstLowerBounds[i] = 0;
                current.firstUpperBounds[i] = 0;
                current.secondLowerBounds[i] = 0;
                current.secondUpperBounds[i] = 0;
            }
        } else {
            // TODO: Wat?
            if(current.firstVP == null) {
                current.firstVP = DBIDUtil.newDistanceDBIDList(tiedFirst);
            }

            // TODO: Wat?
            if(current.secondVP == null || current.secondVP.isEmpty()) {
                // count tied to second vp
                int tiedSecond = 0;
                for(DBIDIter contentIter = content.iter(); contentIter.valid(); contentIter.advance()) {
                    if(DBIDUtil.equal(secondVP, contentIter)) {
                        tiedSecond++;
                    }
                }
                current.secondVP = DBIDUtil.newDistanceDBIDList(tiedSecond);
            }

            int breakPoints = this.kVal - 1;

            double firstDistance;
            double secondDistance;

            ArrayModifiableDBIDs contentArray = DBIDUtil.newArray(content);

            ModifiableDoubleDBIDList[] distances = new ModifiableDoubleDBIDList[this.kVal];

            ModifiableDBIDs[] children = new ModifiableDBIDs[this.kVal];

            for(DBIDArrayMIter iter = contentArray.iter(); iter.valid(); iter.advance()) {
                // If the current position is a VP, this will be set to current
                // offset
                int vpOffset = -1;

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

                int childOffset = -1;

                if (vpOffset == -1){
                    for(int i = 0; i < breakPoints; i++) {
                        int scaleFirstDistance = i + 1;
                        int scaleSecondDistance = this.kVal - scaleFirstDistance;

                        double distanceDiff;

                        // First check for Elements left of Middle
                        // Then switch VP distances to check for right of middle
                        if(scaleFirstDistance <= scaleSecondDistance) {
                            distanceDiff = (scaleSecondDistance * secondDistance) - (scaleFirstDistance * firstDistance);
                        }
                        else {
                            distanceDiff = (scaleFirstDistance * firstDistance) - (scaleSecondDistance * secondDistance);
                        }

                        if(distanceDiff <= 0) {
                            childOffset = i;
                            // TODO: correct? can be true in multiple
                            // partititions
                        }
                    }

                    // If childOffset is still not set, current Object is in
                    // last Partition
                    if(childOffset == -1) {
                        childOffset = this.kVal - 1;
                    }

                    if(children[childOffset] == null) {
                        children[childOffset] = DBIDUtil.newArray();
                    }

                    if(distances[childOffset] == null) {
                        distances[childOffset] = DBIDUtil.newDistanceDBIDList();
                    }

                    if(firstDistance < secondDistance) {
                        distances[childOffset].add(firstDistance,iter);
                    }
                    else {
                        distances[childOffset].add(secondDistance, iter);
                    }

                    children[childOffset].add(iter);
                }

                for(int i = 0; i < this.kVal; i++) {
                    // TODO: how to set bounds correctly?
                    // left, right for even part
                    // firstDist <= secondDist?
                    // check middle part correct in query!
                    if(vpOffset != 0) {
                        current.firstLowerBounds[i] = current.firstLowerBounds[i] > firstDistance ? firstDistance : current.firstLowerBounds[i];
                    }
                    current.firstUpperBounds[i] = current.firstUpperBounds[i] < firstDistance ? firstDistance : current.firstUpperBounds[i];

                    if(vpOffset != 1) {
                        current.secondLowerBounds[i] = current.secondLowerBounds[i] > secondDistance ? secondDistance : current.secondLowerBounds[i];
                    }
                    current.secondUpperBounds[i] = current.secondUpperBounds[i] < secondDistance ? secondDistance : current.secondUpperBounds[i];
                }

            }

            for(int i = 0; i < this.kVal; i++) {
                if(children[i] != null && children.length > 0) {
                    int leftPartititions = i;
                    int rightPartititions = this.kVal - (i + 1);

                    // If Odd amount of partititions, designate middle
                    // partitition as new "Root"
                    if(leftPartititions == rightPartititions) {
                        buildTree(current.childNodes[i] = new Node(this.kVal, ReuseVPIndicator.ROOT), children[i], DBIDUtil.newVar(), DBIDUtil.newDistanceDBIDList());
                    }
                    else if(leftPartititions < rightPartititions) {
                        buildTree(current.childNodes[i] = new Node(this.kVal, ReuseVPIndicator.FIRST_VP), children[i], firstVP, distances[i]);
                    }
                    else {
                        buildTree(current.childNodes[i] = new Node(this.kVal, ReuseVPIndicator.SECOND_VP), children[i], secondVP, distances[i]);
                    }
                }
            }
        }
    }

     private DBIDVarTuple selectVPTuple(DBIDs content){
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
    private DBIDVar selectVPSingle(DBIDs content, DBIDRef vantagePoint){

        switch(this.vpSelector){
        case RANDOM:
            if( content.size() > 1){
                return selectSingleRandomVantagePoint(content);
            }
        case FFT:
            return selectSecondFFTVantagePoint(vantagePoint, content);
        case MAXIMUM_VARIANCE:
            return selectSingleMVVP(content);
        case MAXIMUM_VARIANCE_SAMPLING:
            return selectSampledSingleMVVP(content);
        // Note: since the two Vantage Points with most and second most Variance where selected
        // in the Root Node of the Tree, the pruning quality degresses over recursion and technically the
        // reused VP is not part of context, the second VP is selected by FFT.
        case MAXIMUM_VARIANCE_FFT:
            return selectSecondFFTVantagePoint(vantagePoint,content);
        case MAXIMUM_VARIANCE_FFT_SAMPLING:
            return selectSecondFFTVantagePoint(vantagePoint,content);
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
     * @param content the Set to choose a Vantage Point from
     * @return Vantage Point
     */
    private DBIDVar selectSingleRandomVantagePoint(DBIDs content){
        ArrayModifiableDBIDs contentArray = DBIDUtil.newArray(content);
        DBIDArrayIter contentIter = contentArray.iter();

        int pos = randomThread.nextInt(content.size());
        
        DBIDVar first = DBIDUtil.newVar();
        
        contentIter.seek(pos);
        first.set(contentIter);

        return first;
    }

    /***
     * Selects a Vantage Point from content farthest away from the first one.
     * @param firstVP the first Vantage Point
     * @param content the Set to choose a second Vantage Point from
     * @return a Vantage Point
     */
    private DBIDVar selectSecondFFTVantagePoint(DBIDRef firstVP, DBIDs content){
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
        ArrayModifiableDBIDs workset = DBIDUtil.newArray(content);
        // First VP = random
        DBIDVar first = selectSingleRandomVantagePoint(content);

        // TODO: Really?
        // remove all Datapoints eqaul to first VP from workset
        for(DBIDMIter it = workset.iter(); it.valid(); it.advance()) {
            if(DBIDUtil.equal(it, first)) {
                it.remove();
            }
        }

        // Choose Second VP at Random from remaining DBID's
        DBIDVar second = selectSingleRandomVantagePoint(workset);

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

        // TODO: Truncate!
        if(content.size() == 1) {
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
        double omega = this.mvAlpha * maxDist;

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
     * selects only the single Vantage Point with maximum Variance
     * with no respect to a second VP
     * @param content the set to choose the VP from
     * @return Vantage Point
     */
    private DBIDVar selectSingleMVVP(DBIDs content){
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
     * @param content the Set to select VP from
     * @return Vantage Point
     */
    private DBIDVar selectSampledSingleMVVP(DBIDs content){
        DBIDVar result = DBIDUtil.newVar();

        // TODO: Truncate!
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
        if(this.sampleSize == 2) {
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

        // TODO: Truncate!
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
     * The Node Class saves the important information for the each Node
     * 
     * @author Sebastian Aloisi
     *         based on Node Class for VPTree writtten by Robert Gehde and Erich
     *         Schubert
     */
    protected static class Node {
        /**
         *  Amount of K-splits in this node
         */
        int kVal;

        /**
         * Indicates which VP was reused.
         */
        ReuseVPIndicator vpIndicator;

        /**
         * "Left" Vantage point
         */
        ModifiableDoubleDBIDList firstVP;

        /**
         * "Right" Vantage point
         */
        ModifiableDoubleDBIDList secondVP;

        /**
         * child Trees
         */
        Node[] childNodes;

        /**
         * lower distance bounds
         */
        double[] firstLowerBounds, secondLowerBounds;

        /**
         * upper distance bounds
         */
        double[] firstUpperBounds, secondUpperBounds;
        /**
         * Constructor.
         * 
         */
        public Node(int kval, ReuseVPIndicator vpIndicator) {
            this.kVal = kval;
            this.vpIndicator = vpIndicator;
            firstLowerBounds = new double[this.kVal];
            firstUpperBounds = new double[this.kVal];

            secondLowerBounds = new double[this.kVal];
            secondUpperBounds = new double[this.kVal];
            this.childNodes = new Node[this.kVal];

            for ( int i = 0; i < this.kVal ; i ++){
                this.firstLowerBounds[i] = Double.MAX_VALUE;
                this.secondLowerBounds[i] = Double.MAX_VALUE;

                this.firstUpperBounds[i] = -1;
                this.secondUpperBounds[i] = -1;
            }
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
                        new GHkTreeKNNObjectSearcher() : null;
    }

    @Override
    public KNNSearcher<DBIDRef> kNNByDBID(DistanceQuery<O> distanceQuery, int maxk, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHkRPTreeKNNDBIDSearcher() : null;
    }

    @Override
    public RangeSearcher<O> rangeByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHkTreeRangeObjectSearcher() : null;
    }

    @Override
    public RangeSearcher<DBIDRef> rangeByDBID(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHkTreeRangeDBIDSearcher() : null;
    }

    @Override
    public PrioritySearcher<O> priorityByObject(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHkTreePriorityObjectSearcher() : null;
    }

    @Override
    public PrioritySearcher<DBIDRef> priorityByDBID(DistanceQuery<O> distanceQuery, double maxrange, int flags) {
        return (flags & QueryBuilder.FLAG_PRECOMPUTE) == 0 && //
                distanceQuery.getRelation() == relation && this.distFunc.equals(distanceQuery.getDistance()) ? //
                        new GHkTreePriorityDBIDSearcher() : null;
    }

    /**
     * kNN search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     */
    public static abstract class GHkRPTreeKNNSearcher {
        /**
         * Recursive search function
         * 
         * @param knns Current kNN results
         * @param node Current node
         * @return New tau
         */
        protected double ghkrpKNNSearch(KNNHeap knns, Node node, double reusedDistance) {
            final DBIDs firstVP = node.firstVP;
            final DBIDs secondVP = node.secondVP;

            ReuseVPIndicator vpIndicator = node.vpIndicator;

            double firstVPDistance = 0;

            if(vpIndicator == ReuseVPIndicator.FIRST_VP) {
                firstVPDistance = reusedDistance;
            }
            else {
                // Check knn for vp and singletons
                for(DBIDIter firstVPiter = firstVP.iter(); firstVPiter.valid(); firstVPiter.advance()) {
                    firstVPDistance = queryDistance(firstVPiter);
                    knns.insert(firstVPDistance, firstVPiter);
                }
            }
            
            double tau = knns.getKNNDistance();

            if(secondVP != null) {
                
                double secondVPDistance = 0;

                if(vpIndicator == ReuseVPIndicator.SECOND_VP) {
                    secondVPDistance = reusedDistance;
                }
                else {
                    for(DBIDIter secondVPIter = secondVP.iter(); secondVPIter.valid(); secondVPIter.advance()) {
                        // TODO: only query once?
                        secondVPDistance = queryDistance(secondVPIter);
                        knns.insert(secondVPDistance, secondVPIter);
                    }
                }

                tau = knns.getKNNDistance();

                if(node.childNodes != null) {
                    Node[] children = node.childNodes;

                    // TODO: Priortization
                    for(int i = 0; i < node.kVal; i++) {
                        if(children[i] != null) {
                            int scaleFirstDistance = i + 1;
                            int scaleSecondDistance = node.kVal - scaleFirstDistance;

                            double distanceDiff, lowerBound, upperBound, smallerDistance, currentReuseDistance;

                            // First check for Elements left of Middle
                            // Then switch VP distances to check for right of
                            // middle
                            if(scaleFirstDistance <= scaleSecondDistance) {
                                distanceDiff = ((scaleSecondDistance * secondVPDistance) - (scaleFirstDistance * firstVPDistance)) / 2;
                                lowerBound = node.firstLowerBounds[i];
                                upperBound = node.firstUpperBounds[i];
                                smallerDistance = scaleFirstDistance;
                                currentReuseDistance = firstVPDistance;
                            }
                            else {
                                distanceDiff = ((scaleFirstDistance * firstVPDistance) - (scaleSecondDistance * secondVPDistance)) / 2;
                                lowerBound = node.secondLowerBounds[i];
                                upperBound = node.secondUpperBounds[i];
                                smallerDistance = scaleSecondDistance;
                                currentReuseDistance = secondVPDistance;
                            }

                            // TODO: Bounds correct? range?
                            if(distanceDiff < tau && lowerBound <= smallerDistance + tau && smallerDistance - tau <= upperBound) {
                                tau = ghkrpKNNSearch(knns, children[i], currentReuseDistance);
                            }
                        }
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

    public class GHkTreeKNNObjectSearcher extends GHkRPTreeKNNSearcher implements KNNSearcher<O> {
        private O query;

        @Override
        public KNNList getKNN(O query, int k) {
            final KNNHeap knns = DBIDUtil.newHeap(k);
            this.query = query;
            ghkrpKNNSearch(knns, root, 0);
            return knns.toKNNList();
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHkRPTree.this.distance(query, p);
        }
    }

    /**
     * 
     * kNN search for the GHkRP-Tree
     * 
     * @author Sebastian Aloisi
     */
    public class GHkRPTreeKNNDBIDSearcher extends GHkRPTreeKNNSearcher implements KNNSearcher<DBIDRef> {

        private DBIDRef query;

        @Override
        public KNNList getKNN(DBIDRef query, int k) {
            final KNNHeap knns = DBIDUtil.newHeap(k);
            this.query = query;
            ghkrpKNNSearch(knns, root,0);
            return knns.toKNNList();
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHkRPTree.this.distance(query, p);
        }
    }

    /**
     * Range Searcher for the GH-Tree
     */
    public static abstract class GHkRPTreeRangeSearcher {

        protected void ghkrpRangeSearch(ModifiableDoubleDBIDList result, Node node, double reusedDistance, double range) {
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
                        // TODO: query once?
                        secondVPDistance = queryDistance(secondVPIter);

                        if(secondVPDistance <= range) {
                            result.add(secondVPDistance, secondVPIter);
                        }
                    }
                }

                if(node.childNodes != null) {
                    Node[] children = node.childNodes;

                    for(int i = 0; i < node.kVal; i++) {
                        if(children[i] != null) {
                            int scaleFirstDistance = i + 1;
                            int scaleSecondDistance = node.kVal - scaleFirstDistance;

                            double distanceDiff, upperBound, smallerDistance, currentReuseDistance;

                            // First check for Elements left of Middle
                            // Then switch VP distances to check for right of
                            // middle
                            if(scaleFirstDistance <= scaleSecondDistance) {
                                distanceDiff = ((scaleSecondDistance * secondVPDistance) - (scaleFirstDistance * firstVPDistance)) / 2;
                                upperBound = node.firstUpperBounds[i];
                                smallerDistance = scaleFirstDistance;
                                currentReuseDistance = firstVPDistance;
                            }
                            else {
                                distanceDiff = ((scaleFirstDistance * firstVPDistance) - (scaleSecondDistance * secondVPDistance)) / 2;
                                upperBound = node.secondUpperBounds[i];
                                smallerDistance = scaleSecondDistance;
                                currentReuseDistance = secondVPDistance;
                            }

                            // TODO: Bounds correct? range?
                            if(distanceDiff < range && smallerDistance + range <= upperBound) {
                                ghkrpRangeSearch(result, children[i], currentReuseDistance, range);
                            }
                        }
                    }
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
    public class GHkTreeRangeObjectSearcher extends GHkRPTreeRangeSearcher implements RangeSearcher<O> {

        private O query;

        @Override
        public ModifiableDoubleDBIDList getRange(O query, double range, ModifiableDoubleDBIDList result) {
            this.query = query;
            ghkrpRangeSearch(result, root, 0, range);
            return result;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHkRPTree.this.distance(query, p);
        }
    }

    /**
     * Range search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     */
    public class GHkTreeRangeDBIDSearcher extends GHkRPTreeRangeSearcher implements RangeSearcher<DBIDRef> {

        private DBIDRef query;

        @Override
        public ModifiableDoubleDBIDList getRange(DBIDRef query, double range, ModifiableDoubleDBIDList result) {
            this.query = query;
            ghkrpRangeSearch(result, root, 0, range);
            return result;
        }

        @Override
        protected double queryDistance(DBIDRef p) {
            return GHkRPTree.this.distance(query, p);
        }
    }

    /**
     * Priority search for the GH-Tree
     * 
     * @author Sebastian Aloisi
     * 
     * @param <Q> query type
     */
    public abstract class GHkTreePrioritySearcher<Q> implements PrioritySearcher<Q> {
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

/*             if(cur.node != null) {
                double firstVPDist = queryDistance(cur.node.firstVP);
                Node lc = cur.node.firstChild;

                if(lc != null && intersect(firstVPDist - threshold, firstVPDist + threshold, cur.node.firstLowBound, cur.node.firstHighBound)) {
                    final double mindist = Math.max(firstVPDist - cur.node.firstHighBound, cur.mindist);
                    heap.add(new PrioritySearchBranch(mindist, lc, DBIDUtil.deref(cur.node.firstVP)));
                }

                if(cur.node.secondVP != null) {
                    double secondVPDist = queryDistance(cur.node.secondVP);
                    Node rc = cur.node.secondChild;

                    if(rc != null && intersect(secondVPDist - threshold, secondVPDist + threshold, cur.node.secondLowBound, cur.node.secondHighBound)) {
                        final double mindist = Math.max(secondVPDist - cur.node.secondHighBound, cur.mindist);
                        heap.add(new PrioritySearchBranch(mindist, rc, DBIDUtil.deref(cur.node.secondVP)));
                    }
                }

            } */

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
    public class GHkTreePriorityObjectSearcher extends GHkTreePrioritySearcher<O> {
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
            return GHkRPTree.this.distance(query, p);
        }
    }

    /**
     * Priority Range Search for the GH-tree
     * 
     * @author Sebastian Aloisi
     * 
     */
    public class GHkTreePriorityDBIDSearcher extends GHkTreePrioritySearcher<DBIDRef> {
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
            return GHkRPTree.this.distance(query, p);
        }
    }

    @Override
    public void logStatistics() {
        LOG.statistics(new LongStatistic(this.getClass().getName() + ".distance-computations", distComputations));
    }

    /**
     * Index factory for the GHk-Tree
     * 
     * @author Sebastian Aloisi
     * 
     * @param <O> Object type
     */
    @Alias({ "ghkrp" })
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
         * k-fold parameter
         */
        int kVal;

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
         * @param kVal k-fold parameter
         * @param mvAlpha Maximum Variance threshold
         * @param vpSelector Vantage Point selection Algorithm
         */
        public Factory(Distance<? super O> distance, RandomFactory random, int sampleSize, int truncate, int kval,  double mvAlpha, VPSelectionAlgorithm vpSelector) {
            super();
            this.distance = distance;
            this.random = random;
            this.sampleSize = sampleSize;
            this.truncate = truncate;
            this.kVal = kval;
            this.mvAlpha = mvAlpha;
            this.vpSelector = vpSelector;
        }

        @Override
        public Index instantiate(Relation<O> relation) {
            return new GHkRPTree<>(relation, distance, random, truncate, sampleSize, kVal, mvAlpha, vpSelector);
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
            public final static OptionID DISTANCE_FUNCTION_ID = new OptionID("ghkrptree.distanceFunction", "Distance function to determine the distance between objects");

            /**
             * Parameter to specify the sample size for choosing vantage points
             */
            public final static OptionID SAMPLE_SIZE_ID = new OptionID("ghkrptree.sampleSize", "Size of sample to select vantage points from");

            /**
             * Parameter to specify the minimum leaf size
             */
            public final static OptionID TRUNCATE_ID = new OptionID("ghkrptree.truncate", "Minimum leaf size for stopping");

            /**
             * Parameter specifying k-fold split amount
             */
            public final static OptionID KVAL_ID = new OptionID("ghkrptree.kval", "k-fold parameter");
            
            /**
             * Parameter to specify Maximum Variance Threshold
             */
            public final static OptionID MV_ALPHA_ID = new OptionID("ghkrptree.mvAlpha", "Threshold for Maximum Variance VP selection Algorithm");

            /**
             * Parameter to specify the rnd generator seed
             */
            public final static OptionID SEED_ID = new OptionID("ghkrptree.seed", "The rnd number generator seed");

            /**
             * Parameter to specify the Vantage Point selection Algorithm
             */
            public final static OptionID VPSELECTOR_ID = new OptionID("ghkrptree.vpSelector", "The Vantage Point selection Algorithm");

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
             * k-splits Parameter
             */
            int kVal;

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
                                LOG.warning("GHkRPtree requires a metric to be exact.");
                            }
                        });
                new IntParameter(SAMPLE_SIZE_ID, 10) //
                        .addConstraint(CommonConstraints.GREATER_THAN_ONE_INT) //
                        .grab(config, x -> this.sampleSize = x);
                new IntParameter(TRUNCATE_ID, 8) //
                        .addConstraint(CommonConstraints.GREATER_EQUAL_ONE_INT) //
                        .grab(config, x -> this.truncate = x);
                new IntParameter(KVAL_ID, 2) //
                        .addConstraint(CommonConstraints.GREATER_THAN_ONE_INT) //
                        .grab(config, x -> this.kVal = x);
                new RandomParameter(SEED_ID).grab(config, x -> random = x);
                new DoubleParameter(MV_ALPHA_ID, 0.5) //
                        .addConstraint(CommonConstraints.LESS_THAN_ONE_DOUBLE).addConstraint(CommonConstraints.GREATER_EQUAL_ZERO_DOUBLE).grab(config, x -> this.mvAlpha = x);
                new EnumParameter<>(VPSELECTOR_ID, VPSelectionAlgorithm.class).grab(config, x -> this.vpSelector = x);
            }

            @Override
            public Object make() {
                return new Factory<>(distance, random, sampleSize, truncate, kVal, mvAlpha, vpSelector);
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
                FileWriter fileWriter = new FileWriter("ghkrp.dot");
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

            if(vpIndicator == ReuseVPIndicator.ROOT || vpIndicator != ReuseVPIndicator.FIRST_VP) {
                objectsInNode += firstVP.size();
            }

            if(node.secondVP != null) {
                if(vpIndicator == ReuseVPIndicator.ROOT || vpIndicator != ReuseVPIndicator.SECOND_VP) {
                    objectsInNode += secondVP.size();
                }
            }

            String nodeID = getNodeID(node);

            // TODO: print bounds
            String nodeString = "\"" + nodeID + "\" [ label = \"ID: " + nodeID + "\\n sID: \\n obj: " + String.valueOf(objectsInNode) + "\"]\n";
            this.nodes.add(nodeString);
            this.objectCounter += objectsInNode;

            if(node.childNodes != null) {
                for(int i = 0; i < node.kVal; i++) {
                    if(node.childNodes[i] != null) {
                        String childID = getNodeID(node.childNodes[i]);
                        this.edges.add("\"" + nodeID + "\" -> \"" + childID + "\"\n");
                        parseNode(node.childNodes[i]);
                    }
                }
            }
        }

        private String getNodeID(Node node) {
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
