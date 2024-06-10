package elki.distance;

/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 *
 * Copyright (C) 2024
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

import elki.data.DoubleVector;
import elki.data.NumberVector;
import elki.data.type.SimpleTypeInformation;
import elki.math.linearalgebra.VMath;
import elki.utilities.optionhandling.Parameterizer;

public class GreatCircleDistance implements NumberVectorDistance<DoubleVector>{
    /**
     * Static instance
     */
    public static final GreatCircleDistance STATIC = new GreatCircleDistance();

  /**
   * Constructor - use {@link #STATIC} instead.
   * 
   * @deprecated Use static instance!
   */
  @Deprecated
  public GreatCircleDistance() {
    super();
  }

    @Override
    public double distance(DoubleVector o1, DoubleVector o2) {
        double[] normalO1 = VMath.normalize(o1.toArray());
        double[] normalO2 = VMath.normalize(o2.toArray());
        double dotProduct = VMath.dot(normalO1, normalO2);
        double result = Math.acos(dotProduct);
        return result;
    }

    @Override
    public SimpleTypeInformation<? super NumberVector> getInputTypeRestriction() {
        return NumberVector.FIELD;
    }

    @Override
    public double distance(NumberVector o1, NumberVector o2) {
        // TODO Auto-generated method stub
        return 0;
    }

    @Override
    public boolean isMetric() {
        return true;
    }

    @Override
    public String toString() {
        return "GreatCircleDistance";
    }

    @Override
    public boolean equals(Object obj) {
        return obj == this || (obj != null && this.getClass().equals(obj.getClass()));
    }

    @Override
    public int hashCode() {
        return getClass().hashCode();
    }
    
/**
   * Parameterization class.
   * 
   * @author Sebastian Aloisi
   */
  public static class Par implements Parameterizer {
    @Override
    public GreatCircleDistance make() {
      return GreatCircleDistance.STATIC;
    }
  }
}
