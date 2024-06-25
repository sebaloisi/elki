package elki.distance.strings;

import elki.data.type.SimpleTypeInformation;
import elki.distance.PrimitiveDistance;
import elki.utilities.optionhandling.Parameterizer;

public class NormalizedLongestCommonSubsequence implements PrimitiveDistance<String> {
    /**
     * Static instance, case sensitive.
     */
    public static final NormalizedLongestCommonSubsequence STATIC_SENSITIVE = new NormalizedLongestCommonSubsequence();


    /**
     * Input data type.
     */
    protected static final SimpleTypeInformation<? super String> TYPE = new SimpleTypeInformation<>(String.class);

    /**
   * Constructor. Use static instance instead.
   */
    @Deprecated
     public NormalizedLongestCommonSubsequence() {
        super();
    }


    @Override
    public double distance(String o1, String o2) {
        return normalizedLCS(o1, o2);
    }

    public static double normalizedLCS(String o1, String o2){
        int firstLength = o1.length();
        int secondLength = o2.length();

        if (firstLength == 0 || secondLength == 0){
            return 1;
        }

        double maxLength = Math.max(firstLength, secondLength);

        int[][] steps = new int[firstLength][secondLength];

        for (int i = 1; i < firstLength ; i ++){
            for (int j = 1 ; j < secondLength ; j++){
                if (o1.charAt(i) == o2.charAt(j)){
                    steps[i][j] = steps[i-1][j-1] +1;
                } else {
                    steps[i][j] = Math.max(steps[i][j-1],steps[i-1][j]);
                }
            }
        }

        double lcs = steps[firstLength-1][secondLength-1];
        double normlcs = 1 -(lcs / maxLength);

        return normlcs;
    }

    @Override
    public SimpleTypeInformation<? super String> getInputTypeRestriction() {
        return TYPE;
    }

    @Override
    public boolean isMetric() {
        return true;
    }

    @Override
    public boolean equals(Object obj) {
        return obj == this || (obj != null && this.getClass().equals(obj.getClass()));
    }

    @Override
    public int hashCode() {
        return getClass().hashCode();
    }

    public static class Par implements Parameterizer {
        @Override
        public NormalizedLongestCommonSubsequence make() {
        return STATIC_SENSITIVE;
        }
    }
}
