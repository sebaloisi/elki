package elki.index.tree.metrical.vptree;

import java.util.Random;

import elki.database.ids.DBID;
import elki.database.ids.DBIDIter;
import elki.database.ids.DBIDRef;
import elki.database.ids.DBIDUtil;
import elki.database.ids.DBIDVar;
import elki.database.ids.DoubleDBIDListIter;
import elki.database.ids.DoubleDBIDListMIter;
import elki.database.ids.KNNHeap;
import elki.database.ids.KNNList;
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
import elki.index.tree.metrical.vptree.VPTree.VPTreeKNNDBIDSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreeKNNObjectSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreePriorityDBIDSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreePriorityObjectSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreeRangeDBIDSearcher;
import elki.index.tree.metrical.vptree.VPTree.VPTreeRangeObjectSearcher;
import elki.logging.Logging;
import elki.utilities.documentation.Reference;
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
     * @param vpSelector Vantage Point selection Algorithm
     */
    public GHTree(Relation<O> relation, Distance<? super O> distance, int leafsize, int kVal, VPSelectionAlgorithm vpSelector) {
        this(relation, distance, RandomFactory.DEFAULT, leafsize, leafsize, vpSelector);
    }

    /**
     * Constructor.
     *
     * @param relation data for tree construction
     * @param distance distance function for tree construction
     * @param random Random generator for sampling
     * @param sampleSize Sample size for finding the vantage point
     * @param truncate Leaf size threshold
     * @param vpSelector Vantage Point selection Algorithm
     */
    public GHTree(Relation<O> relation, Distance<? super O> distance, RandomFactory random, int sampleSize, int truncate, VPSelectionAlgorithm vpSelector) {
        this.relation = relation;
        this.distFunc = distance;
        this.random = random;
        this.distQuery = distance.instantiate(relation);
        this.sampleSize = Math.max(sampleSize, 1);
        this.truncate = Math.max(truncate, 1);
        this.vpSelector = vpSelector;
    }

    @Override
    public void initialize() {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'initialize'");
    }

    private enum VPSelectionAlgorithm {
        RANDOM, FFT, MAXIMUM_VARIANCE, MAXIMUM_VARIANCE_SAMPLING, MAXIMUM_VARIANCE_FFT, REF_CHOOSE_VP
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
        // TODO: Variant with max Bounds
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

            // TODO: add multiple selection Methods
            DBIDVarTuple tuple = selectFFTVantagePoints(left, right);

            DBIDVar firstVP = tuple.first;
            DBIDVar secondVP = tuple.second;

            assert !DBIDUtil.equal(firstVP, secondVP);
            int tiedFirst = 0;
            int tiedSecond = 0;
            int firstPartititionSize = 0;
            int secondPartititionSize = 0;

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
                else {
                    secondPartititionSize += 1;
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
            assert DBIDUtil.equal(firstVP, scratchit.seek(right - 1)) : "tiedSecond" + tiedSecond;
            assert (right - left) == firstPartititionSize + secondPartititionSize;

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

            // sort left < 0; right >= 0
            QuickSelectDBIDs.quickSelect(scratch, left + tiedFirst, right, firstPartititionLimit);

            for(scratchit.seek(left + tiedFirst); scratchit.getOffset() < firstPartititionLimit; scratchit.advance()) {
                final double d = scratchit.doubleValue();
                // Move all tied to hyperplane to next Partitition
                if(d == 0) {
                    scratch.swap(scratchit.getOffset(), --firstPartititionLimit);
                    continue;
                }

            }

            for(scratchit.seek(firstPartititionLimit); scratchit.getOffset() < right - tiedSecond; scratchit.advance()) {
                final double d = scratchit.doubleValue();

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
            }
            current.rightChild = buildTree(firstPartititionLimit, right - tiedSecond);

            return current;
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

        private class DBIDVarTuple {
            DBIDVar first;

            DBIDVar second;

            public DBIDVarTuple(DBIDVar first, DBIDVar second) {
                this.first = first;
                this.second = second;
            }
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
        double lowBound, highBound;

        /**
         * Constructor.
         * 
         * @param vp Vantage point and singletons
         */
        public Node(ModifiableDoubleDBIDList leftVp, ModifiableDoubleDBIDList rightVp) {
            this.leftVp = leftVp;
            this.rightVp = rightVp;
            assert !rightVp.isEmpty();
            assert !leftVp.isEmpty();
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
}
